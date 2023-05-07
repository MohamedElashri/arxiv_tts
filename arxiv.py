import subprocess
from pathlib import Path
from typing import List
import numpy as np
import scipy.io.wavfile
from bs4 import BeautifulSoup
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface


def download_paper(paper_id: str) -> None:
    """Download a paper from arXiv."""
    print(f"Downloading paper {paper_id}...")
    cmd = f"arxiv-downloader --id {paper_id} --source"
    subprocess.run(cmd, shell=True, check=True)


def extract_tar_gz(paper_id: str) -> None:
    """Extract the .tar.gz file to a temp folder."""
    tar_gz_file = Path(f"{paper_id}*.tar.gz").resolve().glob().next()
    subprocess.run(["tar", "-xzf", str(tar_gz_file)], check=True)


def get_sentences_from_tex(paper_id: str) -> List[str]:
    """Convert the paper from LaTeX to HTML and extract sentences."""
    # Find the .tex files in the temp folder.
    tex_files = list(Path().resolve().glob("*.tex"))
    # Find the .tex file whose content starts with the string \documentclass.
    documentclass_files = [
        f for f in tex_files
        if f.read_text().startswith("\documentclass")
    ]
    assert len(documentclass_files) == 1, "There should be only one documentclass file."
    documentclass_file = documentclass_files[0]
    # Convert the .tex file to .md file.
    subprocess.run(["pandoc", str(documentclass_file), "-o", f"{paper_id}.html", "-t", "html5"], check=True)
    # Load the .html file with BeautifulSoup4.
    html = Path(f"{paper_id}.html").read_text()
    soup = BeautifulSoup(html, "html.parser")
    # Cleanup. Remove unwanted tags.
    for element in soup(["span", "div", "section", "a", "figure"]):
        if element.get("class") == ["citation"]:
            element.decompose()
        elif element.get("class") in (["math", "inline"], ["math", "display"]):
            element.decompose()
        elif element.get("class") in (["figure"], ["figure*"]):
            element.decompose()
        elif element.get("class") == ["thebibliography"]:
            element.decompose()
        elif element.get("class") == ["center"]:
            element.decompose()
        elif element.get("class") == ["footnotes"]:
            element.decompose()
    # Write the cleaned .html file back.
    Path(f"{paper_id}_cleaned.html").write_text(soup.prettify())
    # Read the cleaned .html file back line by line.
    lines = Path(f"{paper_id}_cleaned.html").read_text().splitlines()
    # Convert to sentences.
    sentences = []
    for line in lines:
        if line.startswith("<"):
            # Opening tags that we expect.
            if line.startswith("<p") or line.startswith("<h1") or line.startswith("<h2") or line.startswith("<h3") or line.startswith("<h4"):
                pass
            # Closing tags that we expect. 
            elif line.startswith("</p>") or line.startswith("</h1>") or line.startswith("</h2>") or line.startswith("</h3>") or line.startswith("</h4>"):
                sentence = accumumlated_sentence.strip().replace("\n", " ")
                # Split by period so that we can insert a pause.
                for x in sentence.split("."):
                    sentences.append(x.strip())
                    sentences.append("<PAUSE>")
                # Start over and add pause.
                accumumlated_sentence = ""
                sentences.append("<PAUSE>")
            else:
                print(f"Unexpected HTML tag: {line}")
        else:
            accumumlated_sentence += line
    # Write to a file.
    sentences = [s for s in sentences if s.strip()]
    Path(f"{paper_id}_sentences.txt").write_text("\n".join(sentences))
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
    # Create a temp directory.
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    # Download the paper.
    download_paper(paper_id)
    # Extract the .tar.gz file to a temp folder.
    extract_tar_gz(paper_id)
    # Convert to sentences.
    sentences = get_sentences_from_tex(paper_id)
    # Convert sentences to speech.
    convert_sentences_to_wav(paper_id, sentences)
    # Remove the temp directory.
    # Remove the temp directory.
    subprocess.run(["rm", "-rf", str(temp_dir)], check=True)


if __name__ == "__main__":
    main()

