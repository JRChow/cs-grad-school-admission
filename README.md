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

For GRE quantitive scores, although the 1st and 3rd quartiles are similar, accepted applicants have a higher median value than the other two types. The same argument applies to GRE verbal scores. Comparing the quantitative and verbal score distributions, we observe that the distribution of GRE quantitative scores is more left-skewed and has a higher median than its verbal distribution, in line with the empirical evidence that most students have a high quantitative GRE score.

In contrast, the GRE writing distribution is obviously more balanced. An interesting discovery is that accepted and rejected applicants share a similar distribution. Both have almost identical minimum, maximum, lower and upper quartiles, with a common median at 4. Additionally, we find that the subset of waitlisted candidates is much smaller than its accepted and rejected counterpart. The summary below confirms our observation - "waitlisted" results constitute around merely 1.48% of the whole dataset. Furthermore, empirical evidence suggests that the majority of waitlisted candidates will be rejected in the end. Considering the above factors, we will treat waitlisted applicants as if they were rejected in remaining parts of this report.

| Category     | `decision` |
| ------------ | ---------- |
| NA:          | 804        |
| Accepted:    | 9971       |
| Rejected:    | 10912      |
| Wait listed: | 326        |
