import pandas as pd


def main():
    cs_raw = pd.read_csv('dataset/cs_raw_with_ranking.csv').dropna()
    print(cs_raw.columns)

    output_df = pd.DataFrame(
            columns=[
                # 'major',
                'is_phd',
                'is_ms',
                'is_spring',
                'is_fall',
                'year',
                'university_name',
                'university_ranking',
                'university_faculty',
                'university_publication',
                'gpa',
                'gre_verbal',
                'gre_quant',
                'gre_writing',
                'is_american',
                'is_international_without_american_degree',
                'is_international_with_american_degree',
                ])

    output_result_df = pd.DataFrame(columns=['decision'])
        
    for index, row in cs_raw.iterrows():
        raw_season = row['season']
        is_spring = True if raw_season[0] == 'F' else False
        is_fall = not is_spring
        year = int(raw_season[1:])
        is_phd = True if row['degree'] == 'PhD' else False
        is_ms = True if row['degree'] == 'MS' or row['degree'] == 'MEng' else False
        if not is_phd and not is_ms:
            print('degree is', row['degree'])
            # raise Exception('is_phd and is_ms cannot both be False!')
        status = row['status']
        is_american = True if status == 'American' else False
        is_international_without_american_degree = True if status != 'American' and status != 'International with US Degree' else False
        is_international_with_american_degree = True if status == 'International with US Degree' else False

        # fill in the data
        output_df.loc[index] = [
            is_phd,
            is_ms,
            is_spring,
            is_fall,
            year,
            row['university'],
            row['world_ranking'],
            row['uni_faculty'],
            row['uni_pub'],
            row['gpa'],
            row['gre_verbal'],
            row['gre_quant'],
            row['gre_writing'],
            is_american,
            is_international_without_american_degree,
            is_international_with_american_degree,
        ]

        # fill in result (accepted or not)
        output_result_df.loc[index] = [True if row['decision'] == 'Accepted' else False]
    
    # export to csv
    output_df.to_csv('dataset/gradcafe/cs_preprocessed_x.csv', index = False)

    output_result_df.to_csv('dataset/gradcafe/cs_preprocessed_y.csv', index = False)


if __name__ == "__main__":
    main()
