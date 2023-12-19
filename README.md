## EMNLP '23: Learning Co-Speech Gesture for Multimodal Aphasia Type Detection 
This codebase contains the python scripts for the model for the EMNLP 2023. https://aclanthology.org/2023.emnlp-main.577/

## Environment & Installation Steps
Python 3.8 & Pytorch 1.12

## Run
Execute the following steps in the same environment:
```
cd Multimodal-Aphasia-Type-Detection_EMNLP_2023 & python main.py
```

## Dataset
- `dataset_chunk50_sample.json`: Due to ethical concerns, we can only provide a sample dataframe consisting of ASR transcripts and labels.

## Ethical Concerns
We sourced the dataset from AphasiaBank with Institutional Review Board (IRB) approval (SKKU2022-11-039) and strictly follow the data sharing guidelines provided by [TalkBank](https://talkbank.org/share/ethics.html), including the Ground Rules for all TalkBank databases based on American Psychological Association Code of Ethics (Association et al., 2002). Additionally, our research adheres to the five core issues outlined by TalkBank in accordance with the General Data Protection Regulation (GDPR) (Regulation, 2018). These issues include addressing commercial purposes, handling scientific data, obtaining informed consent, ensuring deidentification, and maintaining a code of conduct. Considering ethical concerns, we did not use any photographs and personal information of the participants from the AphasiaBank.

---
### If our work was helpful in your research, please kindly cite this work:

```
BIBTEX
@inproceedings{lee2023learning,
  title={Learning Co-Speech Gesture for Multimodal Aphasia Type Detection},
  author={Lee, Daeun and Son, Sejung and Jeon, Hyolim and Kim, Seungbae and Han, Jinyoung},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={9287--9303},
  year={2023}
}
```

### Acknowledgments
This research was supported by the Ministry of Education of the Republic of Korea and the National Research Foundation (NRF) of Korea (NRF2022S1A5A8054322) and the National Research Foundation of Korea grant funded by the Korea government (MSIT) (No. 2023R1A2C2007625).

### Our Lab Site
[Data Science & Artificial Intelligence Laboratory (DSAIL) @ Sungkyunkwan University](https://sites.google.com/view/datasciencelab/home)
