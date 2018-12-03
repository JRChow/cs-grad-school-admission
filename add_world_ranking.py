import pandas as pd


def main():
    # University name 2
    # Major 3
    # Degree 4
    # season (only "S" or "F") 5
    # year (parsed from season) 5
    # university (convert to ranking)
    # Gpa 10
    # Gre_verbal 11
    # Gre_quant 12
    # Gre_writing 13
    # status (American / International / International with US Degree) 16
    # Uni_faculty 20
    # Uni_pub 21
    cs_raw = pd.read_csv('dataset/gradcafe/cs_raw.csv', usecols=[2, 4, 5, 6, 10, 11, 12, 13, 16, 20, 21]).dropna()
    print(cs_raw.columns)
    
    ranking_raw = pd.read_csv('dataset/timesData.csv').dropna()
    print(ranking_raw.columns)

    ranking_df = pd.DataFrame(
            columns=['world_ranking'])

    ranking_dict = dict()
    for index, row in cs_raw.iterrows():
        name = row['university']
        name = name.replace('"', '').lower()
        match_flag = False
        # see if hashtable already contains this university
        if ranking_dict.get(name):
            ranking_df.loc[index] = [ranking_dict[name]]
            continue
        for ranking_index, ranking_row in ranking_raw.iterrows():
            ranking_name = ranking_row['university_name']
            ranking_name = ranking_name.replace('"', '').lower()
            world_rank = ranking_row['world_rank'].replace('=', '')
            if '-' in world_rank:
                original_rank = world_rank
                world_rank = (int(world_rank.split('-')[0]) + int(world_rank.split('-')[1])) // 2
                print('world rank:', original_rank, ' becomes:', world_rank)
            if name in ranking_name or ranking_name in name:
                ranking_dict[name] = world_rank
                print('%s matches %s, rank=%s' % (name, ranking_name, world_rank))
                # fill in the data
                match_flag = True
                ranking_df.loc[index] = [
                    world_rank
                ]
                break
        # if we cannot find a match, manually add world ranking
        if not match_flag:
            print('cannot match university %s, please enter world rank' % name)
            world_rank = input('Enter your input:')
            print('world rank is %s', world_rank)
            ranking_df.loc[index] = [world_rank]
            ranking_dict[name] = world_rank
    
    # export to csv
    cs_raw.join(ranking_df).to_csv('dataset/cs_raw_with_ranking.csv', index = False)

if __name__ == "__main__":
    main()
