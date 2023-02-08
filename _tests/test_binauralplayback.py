import soundfile as sf
import numpy as np
import sys
import simpleaudio as sa

T = 5  # [s] duration of sound

def main():

    dataleft, fs = sf.read('sound_files/speech1.wav')
    dataright, _ = sf.read('sound_files/speech2.wav')

    # Truncate
    dataleft = dataleft[:int(fs * T)]
    dataright = dataright[:int(fs * T)]

    stereoTrack = np.concatenate((
        dataleft[:, np.newaxis],
        dataright[:, np.newaxis]
    ), axis=-1)

    listen_to_array(stereoTrack, fs)


def listen_to_array(audioArray: np.ndarray, fs):
    """Plays back a sounds from an array."""
    # Ensure correct input type and orientation
    if isinstance(audioArray, list):
        audioArray = np.array(audioArray)
    if len(audioArray.squeeze().shape) > 2:
        raise ValueError(f'The provided audio array has too many dimensions (shape: {audioArray.shape}).')
    elif len(audioArray.shape) == 2:  # stereo
        if audioArray.shape[0] < audioArray.shape[1]:
            audioArray = audioArray.T  # transpose if needed
        nChannels = 2
    else:  # mono
        nChannels = 1

    audioArray *= 32767 / np.amax(abs(audioArray), axis=0)
    audioArray = audioArray.astype(np.int16)
    playObj = sa.play_buffer(
        audio_data=audioArray,
        num_channels=nChannels,
        bytes_per_sample=2,
        sample_rate=fs
    )
    playObj.wait_done()


if __name__ == '__main__':
    sys.exit(main())