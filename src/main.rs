use core_foundation::error::CFError;
use core_media_rs::cm_sample_buffer::CMSampleBuffer;
use hound::{WavSpec, WavWriter};
use screencapturekit::{
    shareable_content::SCShareableContent,
    stream::{
        SCStream, configuration::SCStreamConfiguration, content_filter::SCContentFilter,
        output_trait::SCStreamOutputTrait, output_type::SCStreamOutputType,
    },
};

use std::sync::mpsc::Receiver;
use std::{
    collections::HashMap,
    sync::mpsc::{Sender, channel},
    thread,
    time::Duration,
};
use std::os::macos::raw::stat;
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

fn main() -> Result<(), CFError> {
    // Load a context and model.
    let mut context_param = WhisperContextParameters::default();

    // Enable DTW token level timestamp for known model by using model preset
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
        model_preset: whisper_rs::DtwModelPreset::BaseEn,
    };
    // Enable DTW token level timestamp for unknown model by providing custom aheads
    // see details https://github.com/ggerganov/whisper.cpp/pull/1485#discussion_r1519681143
    // values corresponds to ggml-base.en.bin, result will be the same as with DtwModelPreset::BaseEn
    let custom_aheads = [
        (3, 1),
        (4, 2),
        (4, 3),
        (4, 7),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 6),
    ]
    .map(|(n_text_layer, n_head)| whisper_rs::DtwAhead {
        n_text_layer,
        n_head,
    });
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::Custom {
        aheads: &custom_aheads,
    };

    let ctx = WhisperContext::new_with_params(
        "./ggml-tiny.bin",
        context_param,
    )
    .expect("failed to load model");
    // Create a state
    let mut state = ctx.create_state().expect("failed to create key");
    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

    // Edit params as needed.
    // Set the number of threads to use to 1.
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
    let (rx, stream) = dm()?;
    loop {
        let sample = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("could not receive from output_buffer");
        let b = sample.get_audio_buffer_list().expect("should work");
        for buffer_index in 0..b.num_buffers() {
            let buffer = b.get(buffer_index).expect("should work");
            println!(
                "{}: channels={}, size={}",
                buffer_index, buffer.number_channels, buffer.data_bytes_size
            );
            let data_slice = buffer.data();
            let sample_count = data_slice.len() / 4;
            for i in 0..sample_count {
                let sample_bytes = [
                    data_slice[i * 4],
                    data_slice[i * 4 + 1],
                    data_slice[i * 4 + 2],
                    data_slice[i * 4 + 3],
                ];
                let sample_f32 = f32::from_le_bytes(sample_bytes);
                // state
                //     .f(sample_f32)
                //     .expect("failed to feed audio sample");
            }
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
