# TwitterMetrics
Distance Metrics on twitter. Goal is to find look alike accounts. 


I uploaded a few python files that describe how I went about generating thousands of look alike accounts. I'm using optimal transport and Wasserstein distance heavily.  The best results I'm getting is to create a set of 3500 words used by twitter users, chosen so that they predict which tweets the user will retweet.  Then, the put a metric on these words using some diffusion ideas.  With this metric, we can create a Wasserstein distance on users by computing the cost to pair their word distributions.  
It seems to work OK.  
My original goal was to generate lists of accounts of accounts belong to users that live in US and are less likely to vote. (What to do with this information is another question.) Basically this is just sorting out the young people who aren't into politics from the old people. 

As a sample, I generated a list of 1000 users that were close to a user in the demographic.  
