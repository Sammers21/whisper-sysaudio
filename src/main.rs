use core_foundation::error::CFError;
use core_media_rs::cm_sample_buffer::CMSampleBuffer;
use ringbuffer::{AllocRingBuffer, RingBuffer};
use screencapturekit::{
    shareable_content::SCShareableContent,
    stream::{
        SCStream, configuration::SCStreamConfiguration, content_filter::SCContentFilter,
        output_trait::SCStreamOutputTrait, output_type::SCStreamOutputType,
    },
};

use std::sync::mpsc::Receiver;
use std::{
    sync::mpsc::{Sender, channel},
    time::{Duration, Instant},
};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

struct AudioStreamOutput {
    sender: Sender<CMSampleBuffer>,
}

impl SCStreamOutputTrait for AudioStreamOutput {
    fn did_output_sample_buffer(
        &self,
        sample_buffer: CMSampleBuffer,
        _of_type: SCStreamOutputType,
    ) {
        self.sender
            .send(sample_buffer)
            .expect("could not send to output_buffer");
    }
}

fn transcribe_buffer(state: &mut whisper_rs::WhisperState, params: &FullParams, audio: &[f32]) {
    if audio.is_empty() {
        return;
    }

    match state.full(params.clone(), audio) {
        Ok(_) => {
            let num_segments = state.full_n_segments().unwrap_or(0);
            for i in 0..num_segments {
                if let Ok(segment) = state.full_get_segment_text(i) {
                    let text = segment.trim();
                    if !text.is_empty() {
                        println!("TRANSCRIPT: {}", text);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Transcription error: {:?}", e);
        }
    }
}

fn main() -> Result<(), CFError> {
    // Load a context and model.
    let mut context_param = WhisperContextParameters::default();

    // Enable DTW token level timestamp for tiny model
    // Note: Using model preset instead of custom aheads since tiny model has fewer layers
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
        model_preset: whisper_rs::DtwModelPreset::Tiny,
    };

    let ctx = WhisperContext::new_with_params("./ggml-tiny.bin", context_param)
        .expect("failed to load model");
    // Create a state
    let mut state = ctx.create_state().expect("failed to create key");
    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

    // Edit params as needed.
    // Set the number of threads to use to 1.
    let mut params = params;
    params.set_n_threads(10);
    // Enable translation.
    // params.set_translate(true);
    // Set the language to translate to to English.
    // params.set_language(Some("en"));
    // Disable anything that prints to stdout.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Enable token level timestamps
    params.set_token_timestamps(true);

    // Create ring buffer for 500ms at 16KHz sample rate (8000 samples)
    let sample_rate = 16000;
    let buffer_duration_ms = 500;
    let buffer_size = (sample_rate * buffer_duration_ms) / 1000;
    let mut ring_buffer: AllocRingBuffer<f32> = AllocRingBuffer::new(buffer_size);

    // Timer for periodic transcription
    let mut last_transcription = Instant::now();
    let transcription_interval = Duration::from_millis(500);

    println!("Starting audio capture and transcription...");

    let (rx, _stream) = dm()?;
    loop {
        let sample = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("could not receive from output_buffer");
        let b = sample.get_audio_buffer_list().expect("should work");

        for buffer_index in 0..b.num_buffers() {
            let buffer = b.get(buffer_index).expect("should work");

            // Convert audio to mono if it's stereo
            let data_slice = buffer.data();
            let sample_count = data_slice.len() / 4;
            let channels = buffer.number_channels as usize;

            for i in 0..sample_count {
                let sample_bytes = [
                    data_slice[i * 4],
                    data_slice[i * 4 + 1],
                    data_slice[i * 4 + 2],
                    data_slice[i * 4 + 3],
                ];
                let sample_f32 = f32::from_le_bytes(sample_bytes);

                // If stereo, convert to mono by taking left channel only
                // (This assumes interleaved stereo data)
                if channels == 2 && i % 2 == 0 {
                    ring_buffer.push(sample_f32);
                } else if channels == 1 {
                    ring_buffer.push(sample_f32);
                }
            }
        }

        // Check if it's time for transcription
        if last_transcription.elapsed() >= transcription_interval && !ring_buffer.is_empty() {
            let audio_samples: Vec<f32> = ring_buffer.iter().copied().collect();
            transcribe_buffer(&mut state, &params, &audio_samples);
            last_transcription = Instant::now();
        }
    }
}

fn dm() -> Result<(Receiver<CMSampleBuffer>, SCStream), CFError> {
    let (tx, rx) = channel();
    let stream = get_stream(tx)?;
    stream.start_capture()?;
    Ok((rx, stream))
}

fn get_stream(tx: Sender<CMSampleBuffer>) -> Result<SCStream, CFError> {
    let config = SCStreamConfiguration::new().set_captures_audio(true)?;
    let display = SCShareableContent::get().unwrap().displays().remove(0);
    let filter = SCContentFilter::new().with_display_excluding_windows(&display, &[]);
    let mut stream = SCStream::new(&filter, &config);
    stream.add_output_handler(AudioStreamOutput { sender: tx }, SCStreamOutputType::Audio);
    Ok(stream)
}
