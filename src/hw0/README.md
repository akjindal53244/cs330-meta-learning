For problem statement go to `data/CS330_HW0.pdf` file. 

It is a basic example to understand about multitask training (MTL) setup.
1. Task-1: Predict likelihood of a movie for a user (probability).
2. Task-2: Movie Rating prediction (1-5) by a User. It is a regression task.

### Run Model Training
You can experiment with different weights for matrix factorization and regression tasks. Both tasks can optionally share user and item(movie) embeddings and this can be passed as a flag to below training command. Default behaviour is sharing of User and Item embeddings. If you want both tasks to learn task-specific User and Item embeddings pass `--no_shared_embeddings` flag.  

Example 1: Task-1 with 0.99 weight and task-2 with 0.01 weight. Default behaviour of shared User and Item embeddings.
`python main.py --factorization_weight 0.99 --regression_weight 0.01 --logdir run/shared=True_LF=0.99_LR=0.01`

Example 2: Task-1 with 0.5 weight and task-2 with 0.5 weight. both taska do not share User and Item embeddings.
`python main.py --no_shared_embeddings --factorization_weight 0.5 --regression_weight 0.5 --logdir run/shared=False_LF=0.5_LR=0.5`

`logdir` is path for storing summary event files.


# Visualize Model Performance
All the models and corresponding train/eval metrics can be visualized by running `tensorboard --logdir=src/hw1/run`

You can directly clone the repo and launch tensorboard. various runs' event summary files are checked-in. 

I have trained multiple models to understand few things:
1. Pros and cons of embeddings sharing.
2. Train vs eval performance, model overfitting, etc.
3. Effect of different weighting schemes and their performance on both tasks.
4. Do both tasks benefit from each other? Which task benefit more by other task?

![Screenshot from 2023-07-30 11-57-39](https://github.com/akjindal53244/cs330-meta-learning/assets/5215386/79a7f607-6498-4f92-b960-3c9911d06d33)


We have found improvements in MSE loss by clipping model predictions for rating prediction task. This is achieved by adding a sigmoid layer on top of last layer of task-2 and multiplying prediction by 5. This preserves differentiable properties of the model and keeps model prediction in range of [0,5]. This can be set to true by passing `--clip_predicted_rating` flag to model training script.
