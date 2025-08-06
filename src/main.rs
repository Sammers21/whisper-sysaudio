use core_foundation::error::CFError;
use core_media_rs::cm_sample_buffer::CMSampleBuffer;
use hound::{WavSpec, WavWriter};
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

fn sanitize_filename(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | ' ' => c,
            _ => '_',
        })
        .collect::<String>()
        .trim()
        .replace(' ', "_")
        .replace("__", "_")
        .chars()
        .take(50) // Limit to 50 characters
        .collect()
}

fn save_audio_to_wav(
    audio: &[f32],
    filename: &str,
    sample_rate: u32,
    channels: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = WavWriter::create(filename, spec)?;

    for &sample in audio {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}

// Resampling function removed - using original sample rate for better quality
// fn resample_to_16khz(audio: &[f32], original_rate: u32) -> Vec<f32> { ... }

fn transcribe_buffer(
    state: &mut whisper_rs::WhisperState,
    params: &FullParams,
    audio: &[f32],
    sequence: u32,
    original_sample_rate: u32,
    channels: u16,
) {
    if audio.is_empty() {
        return;
    }

    // Convert to mono for transcription if stereo, but keep original for saving
    let mono_audio: Vec<f32> = if channels == 2 {
        audio
            .chunks(2)
            .map(|pair| {
                if pair.len() == 2 {
                    (pair[0] + pair[1]) / 2.0
                } else {
                    pair[0]
                }
            })
            .collect()
    } else {
        audio.to_vec()
    };

    // Use mono audio for transcription
    match state.full(params.clone(), &mono_audio) {
        Ok(_) => {
            let num_segments = state.full_n_segments().unwrap_or(0);
            println!("Segments count: {}", num_segments);

            // Collect all segment texts
            let mut all_text = String::new();
            for i in 0..num_segments {
                if let Ok(segment) = state.full_get_segment_text(i) {
                    let text = segment.trim();
                    if !text.is_empty() {
                        all_text.push_str(text);
                        all_text.push(' ');
                    }
                }
            }

            let final_text = all_text.trim();
            if !final_text.is_empty() {
                println!("TRANSCRIPT: {}", final_text);

                // Create filename with sequence number and transcript
                let sanitized_text = sanitize_filename(final_text);
                let filename = if sanitized_text.is_empty() {
                    format!("{}-unknown.wav", sequence)
                } else {
                    format!("{}-{}.wav", sequence, sanitized_text)
                };

                // Save audio to WAV file (use original audio, not resampled)
                match save_audio_to_wav(audio, &filename, original_sample_rate, channels) {
                    Ok(_) => println!(
                        "Saved audio to: {} ({}Hz, {} channels)",
                        filename, original_sample_rate, channels
                    ),
                    Err(e) => eprintln!("Failed to save audio: {}", e),
                }
            } else {
                // Save even if no meaningful transcript (e.g., for silence/noise)
                let filename = format!("{}-silence.wav", sequence);
                match save_audio_to_wav(audio, &filename, original_sample_rate, channels) {
                    Ok(_) => println!(
                        "Saved silence to: {} ({}Hz, {} channels)",
                        filename, original_sample_rate, channels
                    ),
                    Err(e) => eprintln!("Failed to save audio: {}", e),
                }
            }
        }
        Err(e) => {
            eprintln!("Transcription error: {:?}", e);
            // Still save the audio even if transcription failed
            let filename = format!("{}-error.wav", sequence);
            match save_audio_to_wav(audio, &filename, original_sample_rate, channels) {
                Ok(_) => println!(
                    "Saved error audio to: {} ({}Hz, {} channels)",
                    filename, original_sample_rate, channels
                ),
                Err(e) => eprintln!("Failed to save audio: {}", e),
            }
        }
    }
}

fn main() -> Result<(), CFError> {
    // Load a context and model.
    let mut context_param = WhisperContextParameters::default();

    // Enable DTW token level timestamp for tiny model
    // Note: Using model preset instead of custom aheads since tiny model has fewer layers
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
        model_preset: whisper_rs::DtwModelPreset::LargeV3,
    };

    let ctx = WhisperContext::new_with_params("./ggml-large-v3.bin", context_param)
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
    // // Set the language to translate to English.
    // params.set_language(Some("en"));
    // Disable anything that prints to stdout.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Enable token level timestamps
    params.set_token_timestamps(true);

    // Create ring buffer for 10 seconds - we'll determine the actual sample rate dynamically
    let mut ring_buffer: AllocRingBuffer<f32> = AllocRingBuffer::new(480000); // Large enough for 48kHz * 10s

    // Store the detected sample rate and channel count
    let mut detected_sample_rate: Option<u32> = None;
    let mut detected_channels: Option<u16> = None;

    // Timer for periodic transcription
    let mut last_transcription = Instant::now();
    let transcription_interval = Duration::from_millis(10000);

    // Sequence counter for WAV file naming
    let mut sequence_number = 1u32;

    println!("Starting audio capture and transcription...");

    let (rx, _stream) = dm()?;
    loop {
        let sample = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("could not receive from output_buffer");
        let b = sample.get_audio_buffer_list().expect("should work");

        // Try to detect sample rate from the first buffer if not already detected
        if detected_sample_rate.is_none() {
            // macOS system audio is typically 48kHz for digital audio
            detected_sample_rate = Some(48000);
            println!("Using sample rate: 48000Hz (digital audio standard)");
        }

        for buffer_index in 0..b.num_buffers() {
            let buffer = b.get(buffer_index).expect("should work");

            // Store channel count on first buffer
            if detected_channels.is_none() {
                detected_channels = Some(buffer.number_channels as u16);
                println!("Detected {} channels", buffer.number_channels);
            }

            println!(
                "{}: channels={}, size={}",
                buffer_index, buffer.number_channels, buffer.data_bytes_size
            );

            let data_slice = buffer.data();
            let sample_count = data_slice.len() / 4;

            // Process all samples exactly like the reference code
            for i in 0..sample_count {
                let sample_bytes = [
                    data_slice[i * 4],
                    data_slice[i * 4 + 1],
                    data_slice[i * 4 + 2],
                    data_slice[i * 4 + 3],
                ];
                let sample_f32 = f32::from_le_bytes(sample_bytes);
                ring_buffer.push(sample_f32);
            }
        }

        // Check if it's time for transcription
        if last_transcription.elapsed() >= transcription_interval && !ring_buffer.is_empty() {
            // Create a clean snapshot of the buffer to avoid race conditions
            let buffer_len = ring_buffer.len();
            let mut audio_samples = Vec::with_capacity(buffer_len);

            // Copy samples in correct order (oldest to newest)
            for sample in ring_buffer.iter() {
                audio_samples.push(*sample);
            }

            let sample_rate = detected_sample_rate.unwrap_or(48000);
            let channels = detected_channels.unwrap_or(2);
            println!(
                "Processing {} samples at {}Hz (no resampling, {} channels)",
                audio_samples.len(),
                sample_rate,
                channels
            );

            transcribe_buffer(
                &mut state,
                &params,
                &audio_samples,
                sequence_number,
                sample_rate,
                channels,
            );

            sequence_number += 1;
            last_transcription = Instant::now();

            // Clear the buffer after processing to avoid overlaps
            ring_buffer.clear();
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
