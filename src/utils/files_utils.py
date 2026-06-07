import os
import shutil

from .text_utils import neteja_text


def unificar_audios(audio_folder, output_path):
    os.makedirs(output_path, exist_ok=True)

    contador = 0

    for escola in os.listdir(audio_folder):
        escola_path = os.path.join(audio_folder, escola)

        for classe in os.listdir(escola_path):
            classe_path = os.path.join(escola_path, classe)

            for audio in os.listdir(classe_path):
                old_file = os.path.join(classe_path, audio)

                name, ext = os.path.splitext(audio)

                name_clean = neteja_text(name)
                escola_clean = neteja_text(escola)
                classe_clean = neteja_text(classe)

                new_name = f"{name_clean}_{escola_clean}_{classe_clean}_{contador:04d}{ext}"

                new_file = os.path.join(output_path, new_name)

                shutil.copy2(old_file, new_file)

                contador += 1
