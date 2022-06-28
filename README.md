# SCAMET_RSIC
This is tensorflow 2.2 based repository of SCAMET framework for remote sensing image captioning.
This is official implementation of Spatial-Channel Attention based Memory-guided Transformer (SCAMET) approach.
We have designed encode-decoder based CNN-Transformer approach for describing the multi-spectral, multi-resolution, multi-directional remote sensing images.



## Requirements
- cuda>10
- Tensorflow 2.2
- Matplotlib
- PIL
- NLTK
- Store the remote sensing images of three datasets (Sydney captions, UCM captions and RSICD) from "https://github.com/201528014227051/RSICD_optimal".



## Qualitative Results
- Qualitative analysis shows, proposed SCAMET produces more reliable captions for any kind of remote sensing images than baseline.
<img src="https://user-images.githubusercontent.com/34480222/174289021-c3380b16-0238-4f80-a8c8-65342dc66679.png" width="600" height="800" />

## Attention Heatmap
- Attention heatmap illustrates, the individual ability of spatial and channel wise attention encorporated with CNN for selecting pertinent objects in remote sensing images.
<img src="https://user-images.githubusercontent.com/34480222/174290699-822f3c98-ed44-41e6-a1b1-0de36d966507.png" width="600" height="600" />

## Citation
Our research work is published at "Engineering Appliations of Artificial Intelligence", International scientific journal of Elsevier.

Cite it as:
@article{gajbhiye2022generating,
  title={Generating the captions for remote sensing images: A spatial-channel attention based memory-guided transformer approach},
  author={Gajbhiye, Gaurav O and Nandedkar, Abhijeet V},
  journal={Engineering Applications of Artificial Intelligence},
  volume={114},
  pages={105076},
  year={2022},
  publisher={Elsevier}
}
