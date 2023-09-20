import os
import json
import numpy as np
import scipy as scipy
import scipy.io.wavfile as wav
import scipy.signal
import matplotlib.pyplot as plt
import tensorflow as tf
import pyaudio
from collections import Counter
import time

def waveread_as_pcm16(filename):
    """Read in audio data from a wav file.  Return d, sr."""
    samplerate, wave_data = wav.read(filename)
    # Read in wav file.
    return wave_data, samplerate

def wavread_as_float(filename, target_sample_rate=16000):
    """Read in audio data from a wav file.  Return d, sr."""
    wave_data, samplerate = waveread_as_pcm16(filename)
    desired_length = int(
        round(float(len(wave_data)) / samplerate * target_sample_rate))
    wave_data = scipy.signal.resample(wave_data, desired_length)

    # Normalize short ints to floats in range [-1..1).
    data = np.array(wave_data, np.float32) / 32768.0
    return data, target_sample_rate

def load_labels(file_path):
    label_dict = {}
    with open(file_path, 'r') as f:
        for index, line in enumerate(f.readlines()):
            label_dict[index] = line.strip()
    return label_dict

def inference_non_stream(model_path, label_path, audio_dir):
    # load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # load labels
    label_dict = load_labels(label_path)

    # process files
    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            wav_file = os.path.join(audio_dir, file)
            print("Processing file: ", wav_file)
            # read audio file
            wav_data, samplerate = wavread_as_float(wav_file)
            assert samplerate == 16000
            # plt.plot(wav_data)
            print(len(wav_data))

            ############# EDIT ################
            wav_data = wav_data[:16000]
            # pad input audio with zeros, so that audio len = flags.desired_samples
            padded_wav = np.pad(wav_data, (0, 16000-len(wav_data)), 'constant')
            ############# EDIT ################

            input_data = np.expand_dims(padded_wav, 0)
            input_data.shape
            # set input audio data (by default input data at index 0)
            interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
            # run inference
            interpreter.invoke()
            # get output: classification
            out_tflite = interpreter.get_tensor(output_details[0]['index'])
            out_tflite_argmax = np.argmax(out_tflite)
            print(out_tflite_argmax)
            print(label_path.get(out_tflite_argmax, "INVALID"))

def inference_stream(model_path, label_path, audio_dir):
    # load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # load labels
    label_dict = load_labels(label_path)

    # process files
    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            wav_file = os.path.join(audio_dir, file)
            print("Processing file: ", wav_file)
            # read audio file
            wav_data, samplerate = wavread_as_float(wav_file)
            assert samplerate == 16000
            # plt.plot(wav_data)
            print(len(wav_data))
            lenght_to_trim = len(wav_data) % 320
            print(lenght_to_trim)
            # trim audio data to be a multiple of 320
            wav_data = wav_data[:-lenght_to_trim]
            slice_generator = np.array_split(wav_data, len(wav_data) // 320)

            # for loop with 320 sample slices
            for slice in slice_generator:
                input_data = np.expand_dims(slice, 0)
                print(input_data.shape)
                # set input audio data (by default input data at index 0)
                interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
                # run inference
                interpreter.invoke()
                # get output: classification
                out_tflite = interpreter.get_tensor(output_details[0]['index'])
                out_tflite_argmax = np.argmax(out_tflite)
                print(out_tflite_argmax)
                print(label_path.get(out_tflite_argmax, "INVALID"))

def inference_stream_mic_input(model_path, label_path, in_device, out_device, mic_amp_factor):
    # load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # load labels
    label_dict = load_labels(label_path)
    print(label_dict)

    k = pyaudio.PyAudio()
    out = k.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=16000,
                         output=True,
                         output_device_index=out_device
                         )

    # Create a PyAudio object
    p = pyaudio.PyAudio()
    # Open an audio stream
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, input=True, output=False, input_device_index=in_device)
    # Start the stream
    stream.start_stream()

    buffer = []
    old_label = ""
    # Read audio data from the stream
    while True:
        # Read 320 samples from the stream
        data = stream.read(320)
        # Convert the data to a NumPy array
        samples = np.frombuffer(data, dtype=np.float32)
        #amplify the audio
        samples = samples * mic_amp_factor

        # monitoring of mic input
        ####################
        data = samples.astype(np.float32).tobytes()
        out.write(data)
        ####################
        # Do inference
        input_data = np.expand_dims(samples, 0)
        # set input audio data (by default input data at index 0)
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        # run inference
        interpreter.invoke()
        # get output: classification
        out_tflite = interpreter.get_tensor(output_details[0]['index'])
        out_tflite_argmax = np.argmax(out_tflite)

        buffer.append(label_dict.get(out_tflite_argmax, "INVALID"))
        if len(buffer) > 50:
            buffer.pop(0)
        new_label = Counter(buffer).most_common(1)[0][0]
        if new_label != old_label:
            print(new_label)
            old_label = new_label


    # Stop the stream
    stream.stop_stream()
    # Close the PyAudio object
    p.terminate()
    k.terminate()


def inference_non_stream_mic_input(model_path, label_path, in_device, out_device, mic_amp_factor):
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load labels
    label_dict = load_labels(label_path)
    print(label_dict)

    # Initialize PyAudio
    k = pyaudio.PyAudio()
    out = k.open(format=pyaudio.paFloat32,
                 channels=1,
                 rate=16000,
                 output=True,
                 output_device_index=out_device)

    # Create a PyAudio object
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=16000,
                    input=True,
                    output=False,
                    input_device_index=in_device)

    stream.start_stream()

    sample_buffer = []
    in_speech = False
    sound_level_threshold = 0.02  # Adjust the threshold as needed

    while True:
        data = stream.read(320)
        samples = np.frombuffer(data, dtype=np.float32)
        samples = samples * mic_amp_factor  # Amplify the audio

        # # Monitoring of mic input
        # out.write(samples.astype(np.float32).tobytes())

        # Check for sound level to switch in_speech flag
        sound_level = np.abs(samples).mean()
        
        if not in_speech and sound_level > sound_level_threshold:
            print("detected: ")
            in_speech = True
            sample_buffer = []

        if in_speech:
            sample_buffer.extend(samples)
            # experimental 8000 samples for quicker reaction time
            if len(sample_buffer) >= 8000:
                if len(sample_buffer) < 16000:
                    sample_buffer = np.pad(sample_buffer, (0, 16000 - len(sample_buffer)), 'constant', constant_values=(0, 0))
                # Run inference on the 1-second chunk
                input_data = np.expand_dims(np.array(sample_buffer[:16000]), 0)
                interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
                interpreter.invoke()
                out_tflite = interpreter.get_tensor(output_details[0]['index'])
                out_tflite_argmax = np.argmax(out_tflite)

                print(label_dict.get(out_tflite_argmax, "INVALID"))

                # Reset states
                in_speech = False
                sample_buffer = []

    # Stop the stream
    stream.stop_stream()
    p.terminate()
    k.terminate()


def list_devices():
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    for i in range(0, device_count):
        info = p.get_device_info_by_index(i)
        print("Device {} = {}".format(info["index"], info["name"]))

#main function
if __name__ == '__main__':
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    MODELS_DIR = os.path.join(MAIN_DIR, "models")
    AUDIO_DIR = os.path.join(MAIN_DIR, "audio")
    # straming model paths
    MODEL_PATH_STREAM = os.path.join(MODELS_DIR, 'svdf', "stream_state_internal", "stream_state_internal.tflite")
    LABEL_PATH_STREAM = os.path.join(MODELS_DIR, "svdf", "labels.txt")
    # non streaming model paths
    MODEL_PATH_NON_STREAM = os.path.join(MODELS_DIR, 'kwt3', "tflite_non_stream", "non_stream.tflite")
    LABEL_PATH_NON_STREAM = os.path.join(MODELS_DIR, "kwt3", "labels.txt")

    print(list_devices())
    in_device = int(input("Enter microphone device ID: "))
    out_device = int(input("Enter speaker device ID: "))
    mic_amp_factor = int(input("Enter microphone amplification factor: "))
    model_type = int(input("Enter model type (0: stream / 1: non_stream): "))

    # inference_stream(MODEL_PATH_STREAM, LABEL_PATH_STREAM, AUDIO_DIR)
    # inference_non_stream(MODEL_PATH_NON_STREAM, LABEL_PATH_NON_STREAM, AUDIO_DIR)
    if model_type == 0:
        inference_stream_mic_input(MODEL_PATH_STREAM, LABEL_PATH_STREAM, in_device, out_device, mic_amp_factor)
    elif model_type == 1:
        inference_non_stream_mic_input(MODEL_PATH_NON_STREAM, LABEL_PATH_NON_STREAM, in_device, out_device, mic_amp_factor)
