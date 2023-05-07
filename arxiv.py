import subprocess
from pathlib import Path
from typing import List
import numpy as np
import scipy.io.wavfile
from bs4 import BeautifulSoup
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import sys
import tarfile
def download_paper(paper_id: str) -> None:
    """Download a paper from arXiv."""
    print(f"Downloading paper {paper_id}...")
    cmd = f"arxiv-downloader --id {paper_id} --source"
    subprocess.run(cmd, shell=True, check=True)


def extract_tar_gz(paper_id: str) -> None:
    """Extract the contents of a tar.gz file to a temporary folder."""
    # Find the tar.gz file.
    tar_gz_file = next(Path(".").resolve().glob(f"{paper_id}*.tar.gz"))
    # Extract the tar.gz file to a temp folder.
    with tarfile.open(tar_gz_file) as f:
        f.extractall()

def get_sentences_from_tex(paper_id: str) -> List[str]:
    """Extract sentences from .tex files in a temporary folder."""
    # Find all the .tex files in the temp folder.
    tex_files = list(Path(".").resolve().glob("*.tex"))
    # Find the .tex file whose content starts with the string \documentclass.
    documentclass_files = [
        tex_file for tex_file in tex_files if tex_file.read_text().startswith("\\documentclass")
    ]
    assert len(documentclass_files) == 1, "There should be only one documentclass file."
    documentclass_file = documentclass_files[0]
    # Convert the .tex file to .md file.
    subprocess.run(["pandoc", str(documentclass_file), "-o", f"{paper_id}.html", "-t", "html5"], check=True)
    # Load the .html file with BeautifulSoup4.
    with open(f"{paper_id}.html", "r") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    # Cleanup.
    for element in soup.find_all("span", class_="citation"):
        element.decompose()
    for element in soup.find_all("span", class_="math inline"):
        element.decompose()
    for element in soup.find_all("span", class_="math display"):
        element.decompose()
    for element in soup.find_all("div", class_="figure"):
        element.decompose()
    for element in soup.find_all("div", class_="figure*"):
        element.decompose()
    for element in soup.find_all("div", class_="thebibliography"):
        element.decompose()
    for element in soup.find_all("div", class_="center"):
        element.decompose()
    for element in soup.find_all("section", class_="footnotes"):
        element.decompose()
    for element in soup.find_all("a"):
        element.decompose()
    for element in soup.find_all("figure"):
        element.decompose()
    # Write the .html file back.
    with open(f"{paper_id}_cleaned.html", "w") as f:
        f.write(soup.prettify())
    # Read that .html file back line by line.
    with open(f"{paper_id}_cleaned.html", "r") as f:
        lines = f.readlines()
    # Convert to sentences.
    sentences = []
    accumumlated_sentence = ""
    for line in lines:
        if line.startswith("<"):
            # Opening tags that we expect.
            if line.startswith("<p") or line.startswith("<h1") or line.startswith("<h2") or line.startswith("<h3") or line.startswith("<h4"):
                pass
            # Closing tags that we expect. 
            elif line.startswith("</p>") or line.startswith("</h1>") or line.startswith("</h2>") or line.startswith("</h3>") or line.startswith("</h4>"):
                accumumlated_sentence = accumumlated_sentence.replace("\n", " ")
                # Split by period so that we can insert a pause.
                for x in accumumlated_sentence.split("."):
                    sentences.append(x.strip())
                    sentences.append("<PAUSE>")
                # Start over and add pause.
                accumumlated_sentence = ""
                sentences.append("<PAUSE>")
            else:
                print(f"Unexpected HTML tag: {line}")
        else:
            # Accumulate texts.
            accumumlated_sentence += line
    return sentences


def convert_sentences_to_wav(paper_id: str, sentences: List[str]) -> None:
    """Convert sentences to speech using a TTS model."""
    # Load the TTS model.
    print("Loading TTS model...")
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/fastspeech2-en-ljspeech",
        arg_overrides={"vocoder": "hifigan", "fp16": False}
    )
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(models, cfg)
    # Generate speech.
    print("Generating speech...")
    full_wave_file = []
    rate = 44100
    for text in sentences:
        text = text.strip()
        print(f"Text: \"{text}\"")
        if not text:
            continue
        # Insert a pause.
        if text == "<PAUSE>":
            full_wave_file.extend(np.zeros(rate))
            continue
        # Create the sample.
        sample = TTSHubInterface.get_model_input(task, text)
        wav, rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)
        # Convert wav from torch tensor to numpy array.
        wav = wav.numpy()
        # Append.
        full_wave_file.extend(wav)
    # Convert to numpy array.
    full_wave_file = np.array(full_wave_file, dtype=np.float32)
    # Save the generated audio to a file. 
    wav_path = f"{paper_id}.wav"
    print(f"Saving {wav_path}")
    scipy.io.wavfile.write(wav_path, rate, full_wave_file)
    print("Done.")


def main() -> None:
    # Get the paper ID from the command line. If none is provided work with "Attention is all you need."
    paper_id = "1706.03762"
    if len(sys.argv) > 1:
        paper_id = sys.argv[1]
    # Create a tmp directory.
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    # Download the paper.
    download_paper(paper_id)
    # Extract the .tar.gz file to a tmp folder.
    extract_tar_gz(paper_id)
    # Convert to sentences.
    sentences = get_sentences_from_tex(paper_id)
    # Convert sentences to speech.
    convert_sentences_to_wav(paper_id, sentences)
    # Remove the tmp directory.
    # Remove the tmp directory.
    subprocess.run(["rm", "-rf", str(tmp_dir)], check=True)


if __name__ == "__main__":
    main()

