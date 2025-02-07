INTRODUCTION: 
This is a modified version of the Commonsense Explanations Code
Original code can be found here:
https://github.com/salesforce/cos-e

It was originally made for the paper: "Explain Yourself! Leveraging Language models for Commonsense Reasoning" published in "Proceedings of the Association for Computational Linguistics (ACL2019)"
published in: 2019
available here: https://arxiv.org/abs/1906.02361

By authors: 
Nazneen Rajani Fatema, Bryan McCann, Caiming Xiong, Richard Socher


The aim of this modification is to make it easier to read and use the code provided by the researchers.
We focus on the ver1.0 scripts only, as that was the original focus of the paper.
We also provide attempts to reproduce the results discussed in that paper, although these were not always successful. 


STRUCTURE:
You will find several folders here, all of which have their own readme file to explain how to use the files found there.


SETUP:
To be able to run the code, it is advisable to follow along the steps explained in the author's original readme. 
For ease of use, I will summarize here:
This assumes that you are not completely new to coding and know what python, conda and packages are. 

1.Create a new conda environment using conda create -n cose python=3.7
(Python version may have to be increased if you want to use cuda. My environment was set up with python=3.9 to accommodate the newest version of cuda.)
2.Activate the environment with conda activate cose
(Added: Install the libraries found in the DEPENDENCIES section using conda install <name of library>)
3.Get transformers repository much of this code relies by running git clone https://github.com/huggingface/transformers.git
(You might have to run "conda install git" beforehand)
4.Checkout the huggingface transformers commit that works with the code using: git checkout e14c6b52e37876ee642ffde49367c51b0d374f41

DEPENDENCIES:
You need the following packages: (This list may be incomplete, sadly.) 

-jsonlines
-sacrebleu
-pytorch
-numpy
-pytorch_pretrained_bert (this is best installed using pip install pytorch_pretrained_bert)
-pandas
