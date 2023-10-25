# AmericanStories
 
 The American Stories dataset is a collection of full article texts extracted from historical U.S. newspaper images. It includes nearly 20 million scans from the public domain Chronicling America collection maintained by the Library of Congress. The dataset is designed to address the challenges posed by complex layouts and low OCR quality in existing newspaper datasets.
 It was created using a novel deep learning pipeline that incorporates layout detection, legibility classification, custom OCR, and the association of article texts spanning multiple bounding boxes. It employs efficient architectures specifically designed for mobile phones to ensure high scalability.
 The dataset offers high-quality data that can be utilized for various purposes. It can be used to pre-train large language models and improve their understanding of historical English and world knowledge. 
 The dataset can also be integrated into retrieval-augmented language models, making historical information more accessible, including interpretations of political events and details about people's ancestors.
 Additionally, the structured article texts in the dataset enable the use of transformer-based methods for applications such as detecting reproduced content. This significantly enhances accuracy compared to relying solely on existing OCR techniques.
 The American Stories dataset serves as an invaluable resource for developing multimodal layout analysis models and other multimodal applications. Its vast size and silver quality make it ideal for innovation and research in this domain.
 
 ## Hugging Face Dataset
The dataset is on the [Hugging Face Hub](https://huggingface.co/datasets/dell-research-harvard/AmericanStories). More information about the dataset can be found in the paper and the linked dataset card. 
 
 ## Accessing the data
 Ensure that you have installed the datasets library from Hugging Face.  
 
```
!pip install datasets
 
```
 
 There are 4 configurations possible depending upon the use case. 
 
```
from datasets import load_dataset

#  Download data for the year 1809 at the associated article level (Default)
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "subset_years",
    year_list=["1809", "1810"]
)

# Download and process data for all years at the article level
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "all_years"
)

# Download and process data for 1809 at the scan level
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "subset_years_content_regions",
    year_list=["1809"]
)

# Download ad process data for all years at the scan level
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "all_years_content_regions")

```

## Colab/Jupyter Notebooks

### [Dataset Demo Notebook](https://colab.research.google.com/drive/1ifzTDNDtfrrTy-i7uaq3CALwIWa7GB9A?ts=648b98bf)

### [Dataset Extended Examples Notebook](https://colab.research.google.com/drive/1S5FfPV1vO0fSJl7NPI48dZxoVSSxbkG_?usp=sharing)

### [Processing Scans Example Notebook](https://colab.research.google.com/drive/1eU4M9HUJ1e4r5jnAaNer1VP-hEA35VIm?usp=sharing)

## Replication

We provide all models and scripts used to create American Stories. Processing newspaper scans is relatively simple. Follow the instructions below or refer to the "Processing Scans Example Notebook" above.  

1. Clone this repo to a relevant location and install dependencies:

```
git clone https://github.com/dell-research-harvard/AmericanStories.git`
cd AmericanStories
pip install -r requirements.txt
```

2. Download Models from [this Dropbox Folder](https://www.dropbox.com/sh/sfaf1nmuji9yhu6/AAAj1UGrPmCWFJUiTSP41ihpa?dl=0) to a `american_stories_models` folder

3. Place one or more Newspaper Scans (in .jp2 format) in a `scans` folder. Example scans can be downloaded [here](https://chroniclingamerica.loc.gov/data/batches/ak_albatross_ver01/data/sn84020657/00279526685/1917010301/)

**Note:** PDF format scans are supported, but the dependencies are not install by default, because the `pikepdf` package has caused dependency conflicts on some machines and with some python versions. If you are planning to process pdfs, you can process them in the same way, but must
first install `pikepdf`:

```
pip install pikepdf
```

4. Run `process_scans.sh`, scan output will be saved in an `output` folder. 


