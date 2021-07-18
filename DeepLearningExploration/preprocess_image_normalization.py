import os
import librosa, librosa.display
import matplotlib.pyplot as plt
import os

DATASET_PATH = "genre_dataset"
SAMPLE_RATE = 22050

def save_images(dataset_path):

    # loop through all the genres in the gtzan dataset
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that we are not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split("\\") # genre/blues => ["genre", "blues"]
            semantic_label = dirpath_components[-1] # considering the last index
            print("\nProcessing {}".format(semantic_label))

            # process files for a specific genre
            for j, f in enumerate(filenames):
                # load the audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                librosa.display.waveplot(signal, sr=sr)

                current_folder = "createdImages/{}".format(semantic_label)
                if not os.path.exists(current_folder):
                    os.makedirs(current_folder)

                plt.savefig("{}/{}{}.png".format(current_folder, semantic_label, j))
                plt.clf()

if __name__ == "__main__":
    save_images(DATASET_PATH)