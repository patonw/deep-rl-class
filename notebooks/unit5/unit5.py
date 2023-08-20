# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/notebooks/unit5/unit5.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="2D3NL_e4crQv"
# # Unit 5: An Introduction to ML-Agents
#
#

# %% [markdown] id="97ZiytXEgqIz"
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/thumbnail.png" alt="Thumbnail"/>
#
# In this notebook, you'll learn about ML-Agents and train two agents.
#
# - The first one will learn to **shoot snowballs onto spawning targets**.
# - The second need to press a button to spawn a pyramid, then navigate to the pyramid, knock it over, **and move to the gold brick at the top**. To do that, it will need to explore its environment, and we will use a technique called curiosity.
#
# After that, you'll be able **to watch your agents playing directly on your browser**.
#
# For more information about the certification process, check this section 👉 https://huggingface.co/deep-rl-course/en/unit0/introduction#certification-process

# %% [markdown] id="FMYrDriDujzX"
# ⬇️ Here is an example of what **you will achieve at the end of this unit.** ⬇️
#

# %% [markdown] id="cBmFlh8suma-"
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/pyramids.gif" alt="Pyramids"/>
#
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/snowballtarget.gif" alt="SnowballTarget"/>

# %% [markdown] id="A-cYE0K5iL-w"
# ### 🎮 Environments: 
#
# - [Pyramids](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#pyramids)
# - SnowballTarget
#
# ### 📚 RL-Library: 
#
# - [ML-Agents](https://github.com/Unity-Technologies/ml-agents)
#

# %% [markdown] id="qEhtaFh9i31S"
# We're constantly trying to improve our tutorials, so **if you find some issues in this notebook**, please [open an issue on the GitHub Repo](https://github.com/huggingface/deep-rl-class/issues).

# %% [markdown] id="j7f63r3Yi5vE"
# ## Objectives of this notebook 🏆
#
# At the end of the notebook, you will:
#
# - Understand how works **ML-Agents**, the environment library.
# - Be able to **train agents in Unity Environments**.
#

# %% [markdown] id="viNzVbVaYvY3"
# ## This notebook is from the Deep Reinforcement Learning Course
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/notebooks/deep-rl-course-illustration.jpg" alt="Deep RL Course illustration"/>

# %% [markdown] id="6p5HnEefISCB"
# In this free course, you will:
#
# - 📖 Study Deep Reinforcement Learning in **theory and practice**.
# - 🧑‍💻 Learn to **use famous Deep RL libraries** such as Stable Baselines3, RL Baselines3 Zoo, CleanRL and Sample Factory 2.0.
# - 🤖 Train **agents in unique environments** 
#
# And more check 📚 the syllabus 👉 https://huggingface.co/deep-rl-course/communication/publishing-schedule
#
# Don’t forget to **<a href="http://eepurl.com/ic5ZUD">sign up to the course</a>** (we are collecting your email to be able to **send you the links when each Unit is published and give you information about the challenges and updates).**
#
#
# The best way to keep in touch is to join our discord server to exchange with the community and with us 👉🏻 https://discord.gg/ydHrjt3WP5

# %% [markdown] id="Y-mo_6rXIjRi"
# ## Prerequisites 🏗️
# Before diving into the notebook, you need to:
#
# 🔲 📚 **Study [what is ML-Agents and how it works by reading Unit 5](https://huggingface.co/deep-rl-course/unit5/introduction)**  🤗  

# %% [markdown] id="xYO1uD5Ujgdh"
# # Let's train our agents 🚀
#
# **To validate this hands-on for the certification process, you just need to push your trained models to the Hub**. There’s no results to attain to validate this one. But if you want to get nice results you can try to attain:
#
# - For `Pyramids` : Mean Reward = 1.75
# - For `SnowballTarget` : Mean Reward = 15 or 30 targets hit in an episode.
#

# %% [markdown] id="DssdIjk_8vZE"
# ## Set the GPU 💪
# - To **accelerate the agent's training, we'll use a GPU**. To do that, go to `Runtime > Change Runtime type`
#
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/notebooks/gpu-step1.jpg" alt="GPU Step 1">

# %% [markdown] id="sTfCXHy68xBv"
# - `Hardware Accelerator > GPU`
#
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/notebooks/gpu-step2.jpg" alt="GPU Step 2">

# %% [markdown] id="an3ByrXYQ4iK"
# ## Clone the repository and install the dependencies 🔽
#

# %% id="6WNoL04M7rTa"
# %%capture
# Clone the repository
# !git clone --depth 1 https://github.com/Unity-Technologies/ml-agents

# %% id="d8wmVcMk7xKo"
# %%capture
# Go inside the repository and install the package
# %cd ml-agents
# !pip3 install -e ./ml-agents-envs
# !pip3 install -e ./ml-agents

# %% [markdown] id="R5_7Ptd_kEcG"
# ## SnowballTarget ⛄
#
# If you need a refresher on how this environments work check this section 👉
# https://huggingface.co/deep-rl-course/unit5/snowball-target

# %% [markdown] id="HRY5ufKUKfhI"
# ### Download and move the environment zip file in `./training-envs-executables/linux/`
# - Our environment executable is in a zip file.
# - We need to download it and place it to `./training-envs-executables/linux/`
# - We use a linux executable because we use colab, and colab machines OS is Ubuntu (linux)

# %% id="C9Ls6_6eOKiA"
# Here, we create training-envs-executables and linux
# !mkdir ./training-envs-executables
# !mkdir ./training-envs-executables/linux

# %% [markdown] id="jsoZGxr1MIXY"
# Download the file SnowballTarget.zip from https://drive.google.com/file/d/1YHHLjyj6gaZ3Gemx1hQgqrPgSS2ZhmB5 using `wget`. 
#
# Check out the full solution to download large files from GDrive [here](https://bcrf.biochem.wisc.edu/2021/02/05/download-google-drive-files-using-wget/)

# %% id="QU6gi8CmWhnA"
# !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YHHLjyj6gaZ3Gemx1hQgqrPgSS2ZhmB5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YHHLjyj6gaZ3Gemx1hQgqrPgSS2ZhmB5" -O ./training-envs-executables/linux/SnowballTarget.zip && rm -rf /tmp/cookies.txt

# %% [markdown] id="_LLVaEEK3ayi"
# We unzip the executable.zip file

# %% id="8FPx0an9IAwO"
# %%capture
# !unzip -d ./training-envs-executables/linux/ ./training-envs-executables/linux/SnowballTarget.zip

# %% [markdown] id="nyumV5XfPKzu"
# Make sure your file is accessible 

# %% id="EdFsLJ11JvQf"
# !chmod -R 755 ./training-envs-executables/linux/SnowballTarget

# %% [markdown] id="NAuEq32Mwvtz"
# ### Define the SnowballTarget config file
# - In ML-Agents, you define the **training hyperparameters into config.yaml files.**
#
# There are multiple hyperparameters. To know them better, you should check for each explanation with [the documentation](https://github.com/Unity-Technologies/ml-agents/blob/release_20_docs/docs/Training-Configuration-File.md)
#
#
# So you need to create a `SnowballTarget.yaml` config file in ./content/ml-agents/config/ppo/
#
# We'll give you here a first version of this config (to copy and paste into your `SnowballTarget.yaml file`), **but you should modify it**.
#
# ```
# behaviors:
#   SnowballTarget:
#     trainer_type: ppo
#     summary_freq: 10000
#     keep_checkpoints: 10
#     checkpoint_interval: 50000
#     max_steps: 200000
#     time_horizon: 64
#     threaded: true
#     hyperparameters:
#       learning_rate: 0.0003
#       learning_rate_schedule: linear
#       batch_size: 128
#       buffer_size: 2048
#       beta: 0.005
#       epsilon: 0.2
#       lambd: 0.95
#       num_epoch: 3
#     network_settings:
#       normalize: false
#       hidden_units: 256
#       num_layers: 2
#       vis_encode_type: simple
#     reward_signals:
#       extrinsic:
#         gamma: 0.99
#         strength: 1.0
# ```

# %% [markdown] id="4U3sRH4N4h_l"
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/snowballfight_config1.png" alt="Config SnowballTarget"/>
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/snowballfight_config2.png" alt="Config SnowballTarget"/>

# %% [markdown] id="JJJdo_5AyoGo"
# As an experimentation, you should also try to modify some other hyperparameters. Unity provides very [good documentation explaining each of them here](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-Configuration-File.md).
#
# Now that you've created the config file and understand what most hyperparameters do, we're ready to train our agent 🔥.

# %% [markdown] id="f9fI555bO12v"
# ### Train the agent
#
# To train our agent, we just need to **launch mlagents-learn and select the executable containing the environment.**
#
# We define four parameters:
#
# 1. `mlagents-learn <config>`: the path where the hyperparameter config file is.
# 2. `--env`: where the environment executable is.
# 3. `--run_id`: the name you want to give to your training run id.
# 4. `--no-graphics`: to not launch the visualization during the training.
#
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/mlagentslearn.png" alt="MlAgents learn"/>
#
# Train the model and use the `--resume` flag to continue training in case of interruption. 
#
# > It will fail first time if and when you use `--resume`, try running the block again to bypass the error. 
#
#

# %% [markdown] id="lN32oWF8zPjs"
# The training will take 10 to 35min depending on your config, go take a ☕️you deserve it 🤗.

# %% id="bS-Yh1UdHfzy"
# !mlagents-learn ./config/ppo/SnowballTarget.yaml --env=./training-envs-executables/linux/SnowballTarget/SnowballTarget --run-id="SnowballTarget1" --no-graphics

# %% [markdown] id="5Vue94AzPy1t"
# ### Push the agent to the 🤗 Hub
#
# - Now that we trained our agent, we’re **ready to push it to the Hub to be able to visualize it playing on your browser🔥.**

# %% [markdown] id="izT6FpgNzZ6R"
# To be able to share your model with the community there are three more steps to follow:
#
# 1️⃣ (If it's not already done) create an account to HF ➡ https://huggingface.co/join
#
# 2️⃣ Sign in and then, you need to store your authentication token from the Hugging Face website.
# - Create a new token (https://huggingface.co/settings/tokens) **with write role**
#
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/notebooks/create-token.jpg" alt="Create HF Token">
#
# - Copy the token 
# - Run the cell below and paste the token

# %% id="rKt2vsYoK56o"
from huggingface_hub import notebook_login
notebook_login()

# %% [markdown] id="aSU9qD9_6dem"
# If you don't want to use a Google Colab or a Jupyter Notebook, you need to use this command instead: `huggingface-cli login`

# %% [markdown] id="KK4fPfnczunT"
# Then, we simply need to run `mlagents-push-to-hf`.
#
# And we define 4 parameters:
#
# 1. `--run-id`: the name of the training run id.
# 2. `--local-dir`: where the agent was saved, it’s results/<run_id name>, so in my case results/First Training.
# 3. `--repo-id`: the name of the Hugging Face repo you want to create or update. It’s always <your huggingface username>/<the repo name>
# If the repo does not exist **it will be created automatically**
# 4. `--commit-message`: since HF repos are git repository you need to define a commit message.
#
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/mlagentspushtohub.png" alt="Push to Hub"/>
#
# For instance:
#
# `!mlagents-push-to-hf  --run-id="SnowballTarget1" --local-dir="./results/SnowballTarget1" --repo-id="ThomasSimonini/ppo-SnowballTarget"  --commit-message="First Push"`

# %% id="kAFzVB7OYj_H"
# !mlagents-push-to-hf --run-id="SnowballTarget1" --local-dir="./results/SnowballTarget1" --repo-id="ThomasSimonini/ppo-SnowballTarget" --commit-message="First Push"

# %% id="dGEFAIboLVc6"
# !mlagents-push-to-hf  --run-id= # Add your run id  --local-dir= # Your local dir  --repo-id= # Your repo id  --commit-message= # Your commit message

# %% [markdown] id="yborB0850FTM"
# Else, if everything worked you should have this at the end of the process(but with a different url 😆) :
#
#
#
# ```
# Your model is pushed to the hub. You can view your model here: https://huggingface.co/ThomasSimonini/ppo-SnowballTarget
# ```
#
# It’s the link to your model, it contains a model card that explains how to use it, your Tensorboard and your config file. **What’s awesome is that it’s a git repository, that means you can have different commits, update your repository with a new push etc.**

# %% [markdown] id="5Uaon2cg0NrL"
# But now comes the best: **being able to visualize your agent online 👀.**

# %% [markdown] id="VMc4oOsE0QiZ"
# ### Watch your agent playing 👀
#
# For this step it’s simple:
#
# 1. Remember your repo-id
#
# 2. Go here: https://huggingface.co/spaces/ThomasSimonini/ML-Agents-SnowballTarget
#
# 3. Launch the game and put it in full screen by clicking on the bottom right button
#
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/snowballtarget_load.png" alt="Snowballtarget load"/>

# %% [markdown] id="Djs8c5rR0Z8a"
# 1. In step 1, choose your model repository which is the model id (in my case ThomasSimonini/ppo-SnowballTarget).
#
# 2. In step 2, **choose what model you want to replay**:
#   - I have multiple one, since we saved a model every 500000 timesteps. 
#   - But if I want the more recent I choose `SnowballTarget.onnx`
#
# 👉 What’s nice **is to try with different models step to see the improvement of the agent.**
#
# And don't hesitate to share the best score your agent gets on discord in #rl-i-made-this channel 🔥
#
# Let's now try a harder environment called Pyramids...

# %% [markdown] id="rVMwRi4y_tmx"
# ## Pyramids 🏆
#
# ### Download and move the environment zip file in `./training-envs-executables/linux/`
# - Our environment executable is in a zip file.
# - We need to download it and place it to `./training-envs-executables/linux/`
# - We use a linux executable because we use colab, and colab machines OS is Ubuntu (linux)

# %% [markdown] id="NyqYYkLyAVMK"
# Download the file Pyramids.zip from https://drive.google.com/uc?export=download&id=1UiFNdKlsH0NTu32xV-giYUEVKV4-vc7H using `wget`. Check out the full solution to download large files from GDrive [here](https://bcrf.biochem.wisc.edu/2021/02/05/download-google-drive-files-using-wget/)

# %% id="AxojCsSVAVMP"
# !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UiFNdKlsH0NTu32xV-giYUEVKV4-vc7H' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UiFNdKlsH0NTu32xV-giYUEVKV4-vc7H" -O ./training-envs-executables/linux/Pyramids.zip && rm -rf /tmp/cookies.txt

# %% [markdown] id="bfs6CTJ1AVMP"
# **OR** Download directly to local machine and then drag and drop the file from local machine to `./training-envs-executables/linux`

# %% [markdown] id="H7JmgOwcSSmF"
# Wait for the upload to finish and then run the command below. 
#
# ![image.png](data:image/png;base64,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)

# %% [markdown] id="iWUUcs0_794U"
# Unzip it

# %% id="i2E3K4V2AVMP"
# %%capture
# !unzip -d ./training-envs-executables/linux/ ./training-envs-executables/linux/Pyramids.zip

# %% [markdown] id="KmKYBgHTAVMP"
# Make sure your file is accessible 

# %% id="Im-nwvLPAVMP"
# !chmod -R 755 ./training-envs-executables/linux/Pyramids/Pyramids

# %% [markdown] id="fqceIATXAgih"
# ###  Modify the PyramidsRND config file
# - Contrary to the first environment which was a custom one, **Pyramids was made by the Unity team**.
# - So the PyramidsRND config file already exists and is in ./content/ml-agents/config/ppo/PyramidsRND.yaml
# - You might asked why "RND" in PyramidsRND. RND stands for *random network distillation* it's a way to generate curiosity rewards. If you want to know more on that we wrote an article explaning this technique: https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-random-network-distillation-488ffd8e5938
#
# For this training, we’ll modify one thing:
# - The total training steps hyperparameter is too high since we can hit the benchmark (mean reward = 1.75) in only 1M training steps.
# 👉 To do that, we go to config/ppo/PyramidsRND.yaml,**and modify these to max_steps to 1000000.**
#
# <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit7/pyramids-config.png" alt="Pyramids config"/>

# %% [markdown] id="RI-5aPL7BWVk"
# As an experimentation, you should also try to modify some other hyperparameters, Unity provides a very [good documentation explaining each of them here](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-Configuration-File.md).
#
# We’re now ready to train our agent 🔥.

# %% [markdown] id="s5hr1rvIBdZH"
# ### Train the agent
#
# The training will take 30 to 45min depending on your machine, go take a ☕️you deserve it 🤗.

# %% id="fXi4-IaHBhqD"
# !mlagents-learn ./config/ppo/PyramidsRND.yaml --env=./training-envs-executables/linux/Pyramids/Pyramids --run-id="Pyramids Training" --no-graphics

# %% [markdown] id="txonKxuSByut"
# ### Push the agent to the 🤗 Hub
#
# - Now that we trained our agent, we’re **ready to push it to the Hub to be able to visualize it playing on your browser🔥.**

# %% id="yiEQbv7rB4mU"
# !mlagents-push-to-hf  --run-id= # Add your run id  --local-dir= # Your local dir  --repo-id= # Your repo id  --commit-message= # Your commit message

# %% [markdown] id="7aZfgxo-CDeQ"
# ### Watch your agent playing 👀
#
# 👉 https://huggingface.co/spaces/unity/ML-Agents-Pyramids

# %% [markdown] id="hGG_oq2n0wjB"
# ### 🎁 Bonus: Why not train on another environment?
# Now that you know how to train an agent using MLAgents, **why not try another environment?** 
#
# MLAgents provides 17 different and we’re building some custom ones. The best way to learn is to try things of your own, have fun.
#
#

# %% [markdown] id="KSAkJxSr0z6-"
# ![cover](https://miro.medium.com/max/1400/0*xERdThTRRM2k_U9f.png)

# %% [markdown] id="YiyF4FX-04JB"
# You have the full list of the Unity official environments here 👉 https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Examples.md
#
# For the demos to visualize your agent 👉 https://huggingface.co/unity
#
# For now we have integrated: 
# - [Worm](https://huggingface.co/spaces/unity/ML-Agents-Worm) demo where you teach a **worm to crawl**.
# - [Walker](https://huggingface.co/spaces/unity/ML-Agents-Walker) demo where you teach an agent **to walk towards a goal**.

# %% [markdown] id="PI6dPWmh064H"
# That’s all for today. Congrats on finishing this tutorial!
#
# The best way to learn is to practice and try stuff. Why not try another environment? ML-Agents has 17 different environments, but you can also create your own? Check the documentation and have fun!
#
# See you on Unit 6 🔥,
#
# ## Keep Learning, Stay  awesome 🤗
