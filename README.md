# ReID_framework

Person Re-ID framework in development during my PhD with the collaboration of Prof. Dr. William Robson Schwartz (UFMG).
 
In this framework we have implemented the following works:
 
<p><a href="http://www.ssig.dcc.ufmg.br/reid-results/"><strong>Click here</strong></a> for a comprehensive list with&nbsp;results for several person re-identification datasets.</p>
<p>&nbsp;</p>
<table style="height: 217px;" width="525">
<tbody>
<tr>
<td>
<ol>
<li><strong>Raphael Prates, William Robson Schwartz, &ldquo;CBRA &ndash; Color-Based Ranking Aggregation for Person Re-Identification,&rdquo; in IEEE ICIP, 2015.</strong></li>
<li style="text-align: justify;"><strong>Raphael Prates, William Robson Schwartz, &ldquo;Appearance-Based Person Re-identification by Intra-Camera Discriminative Models and Rank Aggregation,&rdquo; in International Conference on Biometrics (ICB), 2015.</strong></li>
<li style="text-align: justify;"><strong>Raphael Prates, Cristiane Dutra and William R. Schwartz, &ldquo;PREDOMINANT COLOR NAME INDEXING STRUCTURE FOR PERSON RE-IDENTIFICATION,&rdquo; in IEEE ICIP, 2016.</strong></li>
<li style="text-align: justify;"><strong>Raphael Prates, Marina Oliveira and William Robson Schwartz, &ldquo;Kernel Partial Least Squares for Person Re-Identification,&rdquo; in IEEE AVSS, 2016.</strong></li>
<li><strong>Raphael Prates and William Robson Schwartz, &ldquo;Kernel Hierarchical PCA for Person Re-Identification,&rdquo; in ICPR, 2016.</strong></li>
</ol>
</td>
</tr>
</tbody>
</table>
Raphael Prates and William Robson Schwartz, “Kernel Hierarchical PCA for Person Re-Identification,” in ICPR, 2016.

@inproceedings{Prates2016ICPR,

title = {Kernel Hierarchical PCA for Person Re-Identification},

author = {Raphael Prates and William Robson Schwartz},

url = {http://www.ssig.dcc.ufmg.br/wp-content/uploads/2016/07/kernelHPCA.pdf},

year = {2016},

date = {2016-12-13},

booktitle = {23th International Conference on Pattern Recognition, ICPR 2016, Cancun, MEXICO, December 4-8, 2016.}

}

Raphael Prates, Marina Oliveira and William Robson Schwartz, “Kernel Partial Least Squares for Person Re-Identification,” in IEEE AVSS, 2016.

@inproceedings{Prates2016AVSS,

title = {Kernel Partial Least Squares for Person Re-Identification},

author = {Raphael Prates and Marina Oliveira and William Robson Schwartz},

url = {http://www.ssig.dcc.ufmg.br/wp-content/uploads/2016/07/egpaper_for_DoubleBlindReview.pdf},

year = {2016},

date = {2016-08-24},

booktitle = {IEEE International Conference on Advanced Video and Signal-Based Surveillance (AVSS)}

}

Raphael Prates, Cristiane Dutra and William R. Schwartz, “Predominant Color Name Indexing Structure for Person Re-Identification,” in IEEE ICIP, 2016.

@inproceedings{Prates:2015:ICIP2015,

title = {Predominant Color Names Indexing Structure for Person Re-Identification},

author = {R. F. de C. Prates, Dutra, C. R. S. and W. R. Schwartz},

year = {2016},

booktitle = {IEEE International Conference on Image Processing (ICIP)},

pages = {1-5},

}

Raphael Prates, William R. Schwartz, “CBRA – Color-Based Ranking Aggregation for Person Re-Identification,” in IEEE ICIP, 2015

@inproceedings{Prates:2015:ICIP2015,

title = {CBRA: Color-Based Ranking Aggregation for Person Re-Identification},

author = {R. F. de C. Prates and W. R. Schwartz},

url = {http://www.ssig.dcc.ufmg.br/wp-content/uploads/2015/06/paper_2015_ICIP_Prates.pdf},

year = {2015},

booktitle = {IEEE International Conference on Image Processing (ICIP)},

pages = {1-5}

}

Raphael Prates, William R. Schwartz, “Appearance-Based Person Re-identification by Intra-Camera Discriminative Models and Rank Aggregation,” in International Conference on Biometrics (ICB), 2015.

@inproceedings{Prates:2015:ICBb,

title = {Appearance-Based Person Re-identification by Intra-Camera Discriminative Models and Rank Aggregation},

author = author = {R. F. de C. Prates and W. R. Schwartz},

url = {http://www.ssig.dcc.ufmg.br/wp-content/uploads/2015/06/paper_2015_ICB_Prates.pdf},

year = {2015},

booktitle = {International Conference on Biometrics},

series = {Lecture Notes in Computer Science},

pages = {1-8}

}

Instructions:

In order to reproduce the experiments you need to download the respective datasets:

VIPER (https://vision.soe.ucsc.edu/node/178) - You need to include VIPER in a folder (.\datasets\viper\camX) where X is equal A or B for cameras A and B, respectively.

PRID450S (http://lrs.icg.tugraz.at/download.php) - You need to include PRID450S in a folder (.\datasets\prid450s\camX) where X is equal A or B for cameras A and B, respectively.

You can run all the experiments from ReID_framework.m by uncommenting the respective lines in the file ReID_framework.m. The final results are stored in the Folder (./Graphics).

Notice that in AVSS2016 and ICPR2016 we used the features proposed in the paper (which are available in the folder .\auxiliary). Therefore, if you use these features, please cite their work:

"An Enhanced Deep Feature Representation for Person Re-identification" . Shangxuan Wu, Ying-Cong Chen, Xiang Li, Jin-Jie You, Wei-Shi Zheng. IEEE Winter Conference on Applications of Computer Vision (WACV2016).

Any question, please contact us:

pratesufop@gmail.com (Raphael Prates).
 
 
