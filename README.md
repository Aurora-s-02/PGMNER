# Named Entity Recognition and Coreference Resolution Using Prompt-based Generative Multimodal Techniques
In this paper, we focus on multimodal co-reference disambiguation and named entity recognition. Aiming at the problems of incomplete understanding of context and inter-modal data interaction in named entity recognition, we propose a generative multimodal named entity recognition model based on prompt hints.The model effectively improves the model's ability to process and integrate semantic information between different modalities by fusing the target detection and prompt cueing techniques.

# Dependencies
This code requires the following:python 3.8

# Prepare dataset
Create a folder.data/
Data is [here](https://drive.google.com/drive/folders/1qiMRSq4xgaZhOL6AfDOvYr_yHyXuY7QU?usp=drive_link).

# Training script
To save the models create a folder and then run the training script below for the final model. output/
```--dataset_type=data/
--context_representation=decoder
--role_path=./data/dset_meta/
--prompt_path=./data/prompts/
--seed=41
--output_dir=output/
--learning_rate=2e-5
--batch_size=8
--max_steps=10000
--use_prompt
--prompt_len=4
--max_enc_seq_length
500
--max_dec_seq_length
200
--window_size
260
--bipartite
--warmup_steps
0.1
--num_prompt_pos
10 ```

