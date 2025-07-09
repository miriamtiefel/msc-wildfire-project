# Dataset Structure

This dataset is organized into three main subsets:

- **train/**
- **val/**
- **test/**

Each subset contains video sequences of fire incidents. The structure within each subset is as follows:

```
subset_name/
├── video1/
│   ├── labels/
│   │   ├── frame2.txt
│   │   └── ...
│   ├── frame1.jpg
│   ├── frame2.jpg
│   ├── frame3.jpg

├── video2/
│   ├── labels/
│   ├── frame1.jpg
│   ├── frame2.jpg
│   └── ...
└── ...
```

### File Details

1. **labels/**: Contains text files corresponding to each video frame. Each file specifies the bounding boxes where fires are detected within the corresponding frame. The format for each bounding box is:
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
   - `class_id`: Class label for the object (e.g., fire).
   - `x_center`, `y_center`: Normalized coordinates of the bounding box center.
   - `width`, `height`: Normalized width and height of the bounding box.

2. **frames**: Contains the individual image frames extracted from the video. The frames are named based on the tower and date in the format: `hpwren_figlib_rmnmobocX01000_2016_06_04T20_00_00.jpg`. These names provide contextual information about the tower and timestamp of each frame. 

3. **metadata.json**: Contains the metadata for the folders, with information such as:
   ```json
   {
       "video": "20160604_FIRE_rm-n-mobo-c",
       "source": "hpwren",
       "dataset": "train",
   }
   ```
   - `video`: Identifier for the video sequence.
   - `source`: Indicates the origin or provider of the data for the frame. Sources can include figlib, pyronear, awf, or adf
   - `dataset`: Specifies the subset (train, val, test) to which the frame belongs.

### Data Sources

The dataset is compiled from various sources to ensure diversity and quality:

- **ALERTWildFire USA**: [ALERTWildfire is a consortium of The University of Nevada, Reno, and the University of Oregon providing fire cameras and tools to help firefighters and first responders.](https://www.alertwildfire.org)
- **FigLib USA**: [A curated library of fire imagery from various US-based events.](https://www.hpwren.ucsd.edu/FIgLib/)
- **PyroNear (2024)**: [Dataset collected in 2024 from fire monitoring systems in France and Spain; internal non-public repository.](https://pyronear.org/es/)
- **PyroNear-sdis (public, 2025)**: [Public dataset collected by Pyronear in 2025 in France and Spain, published as “pyro-sdis” on Hugging Face.](https://huggingface.co/datasets/pyronear/pyro-sdis)
- **Fairefighter  Chile**: [Dataset collected in Chile through a collaborative effort with Pyronear as part of a joint research project.](https://www.chilewebpage.com)
- **Internet from Google**: Random images obtained from online sources.
- **Synthetic**: Images generated in-house by PyroNear to simulate fire incidents and expand the dataset.

### Another Datasets

For comparison, the following images datasets are available:

- **SmokeFrames**: [Download here](https://drive.google.com/file/d/1xtVfJWRaoVXwYjJFADQPRDukufgBHhIV/view?usp=drive_link)  

- **AiForMankind**: [Download here](https://drive.google.com/file/d/1kXzF--BmVUNBdG2EHCF3jDb5QmYaj3EB/view?usp=drive_link)  

- **2019a-smoke-full**: [Download here](https://drive.google.com/file/d/11cU3DYDVtRPjIYLyT345y2L-wCOYagCK/view?usp=drive_link)  

- **NEMO**: [Repository on GitHub](https://github.com/SayBender/Nemo)  

Additionally, there is a **video-based dataset**:

- **SmokeNet ProjectX**: [Video dataset available here](https://archive.org/details/smokenet-projectx)  
  This dataset contains video sequences of fire and smoke events but has limited annotation quality.


### Processing for Replication

#### Image Training

To replicate the Image Dataset, follow these steps:

1. **Cleaning**: Remove frames that contain nighttime.

2. **Dataset Strategy**:
   - For non-fire sequences, select one image per sequence.
   - For fire sequences:
     - Include the first detection of fire.
     - Select additional images approximately evenly across the duration of the fire, aiming for around 6 images per incident.

3. **Training, Validation, and Test**:
   - A pre-trained YOLO-S model was used to train and predict the locations of bounding boxes.
   - The validation set was used to find the best set of parameters to optimize the F1-score, specifically the confidence threshold for predictions.
   - Finally, the test set was used to obtain the final metrics.

#### Video Training

1. **Cleaning**: Remove frames that contain nighttime fire incidents.

2. **Training, Validation, and Test**:
   - Using the previous YOLO training from the Image Dataset, generate possible detections. These detections are passed through an LSTM that classifies the frames based on the detection frame and its preceding frames.
   - The confidence threshold of YOLO is lowered to capture a higher number of potential fires, with the LSTM classifier given more importance in determining the final results.
   - For the test set, we evaluate how many frames detect fire from the moment the fire begins.

### Related Work

#### References

- **AIforMankind [2023]**  
  AIforMankind. AI for Mankind, 2023.  
  [URL](https://aiformankind.org/)

- **ALERTWildfire [2023]**  
  ALERTWildfire. ALERT Wildfire, 2023.  
  [URL](https://www.alertwildfire.org/)

- **HPWREN [2023]**  
  HPWREN. High Performance Wireless Research & Education Network, 2023.  
  [URL](http://hpwren.ucsd.edu/cameras/)


### Our Work

#### Reference

- **Scrapping The Web For Early Wildfire Detection: A New Annotated Dataset of Images and Videos of Smoke Plumes In-the-wild**  
  arXiv:2402.05349v2 [cs.CV], 22 Nov 2024.  
  **Mateo Lostanlen**  
  PyroNear, Paris, France  
  [mateo@pyronear.org](mailto:mateo@pyronear.org)  

  **Core Contributors**:  
  - **Nicolas Isla**  
    Universidad de Chile | CENIA, Santiago, Chile  
    [nicolas.isla@ug.uchile.cl](mailto:nicolas.isla@ug.uchile.cl)  
  - **Renzo Zanca**
    Universidad de Chile | CENIA, Santiago, Chile  
    [nicolas.isla@ug.uchile.cl](mailto:renzo.zanca@ug.uchile.cl) 
  - **Jose Guillen**  
    CENIA, Santiago, Chile  
    [jose.guillen@cenia.cl](mailto:jose.guillen@cenia.cl)  
  - **Felix Veith**  
    PyroNear, Paris, France  
    [felix@pyronear.org](mailto:felix@pyronear.org)  
  - **Cristian Buc**  
    CENIA, Santiago, Chile  
    [cristan.buc@cenia.cl](mailto:cristan.buc@cenia.cl)  
  - **Valentin Barriere**  
    DCC – Universidad de Chile | CENIA, Santiago, Chile  
    [vbarriere@dcc.uchile.cl](mailto:vbarriere@dcc.uchile.cl)















