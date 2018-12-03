# CS Graduate School Admission: Analysis and Prediction
Analyzing historical CS graduate admission records in the US from 2009 Fall to 2016 Fall and try to make predictions.

## Contributors
[Boxuan (Robert) Li](https://www.li-boxuan.com/), [Wenxu (Mike) Mao](http://mike-mao.com/), and [Jingran (Jerome) Zhou](http://jingran-zhou.com/), all contributed equally.

## Questions to Which We Seek Answers
- What are the factors that are related to admission results?
- Is there a single dominant factor in the context of admission?
- Does GPA or GRE scores matter?
- How can we predict the admission result?
- Are there any interesting phenomena that can be found within admission data?

## Dataset: `TheGradCafe`
The dataset we use is the [TheGradCafe](https://github.com/deedy/gradcafe_data). We utilize the subset of computer science admission data of 27,822 results.

## Interesting Findings
### Skewness of GRE score distributions
**GRE verbal**, **GRE quantatitive**, and **GRE writing** all follow a left-skewed distribution. `gre_quant` is probably the most skewed, while `gre_writing` has the least skewness among the three.

![The distribution of GRE quantitative, verbal and writing scores](img/gre.png "The distribution of GRE quantitative, verbal and writing scores")

For GRE quantitive scores, although the 1^st^ and 3^rd^ quartiles are similar, accepted applicants have a higher median value than the other two types. The same argument applies to GRE verbal scores. Comparing the quantitative and verbal score distributions, we observe that the distribution of GRE quantitative scores is more left-skewed and has a higher median than its verbal distribution, in line with the empirical evidence that most students have a high quantitative GRE score.
