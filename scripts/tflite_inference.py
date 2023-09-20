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

def index_to_label(index):
    label_dict = {
        0: '_silence_',
        1: '_unknown_',
        2: 'wow',
        3: 'dog',
        4: 'happy',
        5: 'go',
        6: 'stop',
        7: 'house',
        8: 'marvin',
        9: 'sheila',
        10: 'cat',
        11: 'follow',
    }
    return label_dict.get(index, "Label not found")

def inference_non_stream(model_path, audio_dir):
    # load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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
            print(index_to_label(out_tflite_argmax))

def inference_stream(model_path, audio_dir):
    # load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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
                print(index_to_label(out_tflite_argmax))

def inference_stream_mic_input(model_path, audio_device):
    # load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    k = pyaudio.PyAudio()
    out = k.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=16000,
                         output=True,
                         output_device_index=4
                         )

    # Create a PyAudio object
    p = pyaudio.PyAudio()
    # Open an audio stream
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, input=True, output=False, input_device_index=audio_device)
    # Start the stream
    stream.start_stream()

    time_start = time.time()
    buffer = []
    # Read audio data from the stream
    while True:
        # Read 320 samples from the stream
        data = stream.read(320)
        # Convert the data to a NumPy array
        samples = np.frombuffer(data, dtype=np.float32)
        #amplify the audio
        samples = samples *2

        ####################
        data = samples.astype(np.float32).tobytes()
        out.write(data)
        ####################
        # Do inference
        input_data = np.expand_dims(samples, 0)
        # print(input_data.shape)
        # set input audio data (by default input data at index 0)
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        # run inference
        interpreter.invoke()
        # get output: classification
        out_tflite = interpreter.get_tensor(output_details[0]['index'])
        out_tflite_argmax = np.argmax(out_tflite)

        buffer.append(index_to_label(out_tflite_argmax))
        if len(buffer) > 20:
            buffer.pop(0)
        print(Counter(buffer).most_common(1)[0][0])


    # Stop the stream
    stream.stop_stream()
    # Close the PyAudio object
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
    curr_dir = os.getcwd()
    MODEL_PATH_NON_STREAM = os.path.join(curr_dir, 'svdf', "tflite_non_stream", "non_stream.tflite")
    MODEL_PATH_STREAM = os.path.join(curr_dir, 'svdf', "stream_state_internal", "model.tflite")
    LABEL_PATH = os.path.join(curr_dir, "svdf", "labels.txt")
    AUDIO_DIR = os.path.join(curr_dir, "audio")

#   inference_non_stream(MODEL_PATH_NON_STREAM, AUDIO_DIR)
#   inference_stream(MODEL_PATH_STREAM, AUDIO_DIR)
    # list_devices()
    inference_stream_mic_input(MODEL_PATH_STREAM, audio_device=2)

