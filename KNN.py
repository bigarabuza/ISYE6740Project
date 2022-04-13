#%%
import pandas as pd
from sklearn.neighbors import NearestNeighbors

#%%
#Read The Data
game_ratings = pd.read_csv('games_rating.csv')
user_ratings = pd.read_csv('usergamesrating.csv')
game_ratings['genre'] = game_ratings.genre.astype('category')

#Get User IDs
user_data = user_ratings['steamid'].unique()

#%% 
#Dummy Variables for Genre
game_genres = game_ratings['genre'].unique().tolist()
del game_genres[12]
genre_dummies = pd.get_dummies(game_ratings['genre'])
game_ratings1 = pd.concat([game_ratings, genre_dummies], axis=1) 
game_ratings1 = game_ratings1.drop(columns=['genre'])

#%%
def game_recommendation(user_id):
    user_games = user_ratings.loc[user_ratings['steamid'] == user_id]
    user_games = user_games.sort_values(by=['rating'], ascending=False)
    user_appid = user_games['appid'].iat[0]
    your_game = game_ratings1[game_ratings1['appid'] == user_appid]
    your_game = your_game.iat[0,1]
    
    user_top_game = game_ratings1.loc[game_ratings1['appid'] == user_appid]
    user_top_game = user_top_game.drop(columns=['appid'])
    game_ratings2 = game_ratings1.drop(columns=['appid'])
    
    data_train = game_ratings2.drop(columns=['title'])
    data_test = user_top_game.drop(columns=['title'])
    
    model = NearestNeighbors(n_neighbors=10, metric='manhattan', p=1).fit(data_train)
    recs = model.kneighbors(data_test, 10, return_distance=True)  
    rec_ind = recs[1][0]
    rec_ind = rec_ind.tolist()
    
    rec_list = []
    for ind in rec_ind:
        game_title = game_ratings2.loc[[ind],['title']]
        rec_list.append(game_title.iat[0,0])
        
    #Filter Out Games
    user_id = user_data[0]
    played_games = user_ratings[user_ratings['steamid']== user_id]
    played_games = played_games['appid']

    appid_list = []
    num_games = len(played_games)
    for i in range (num_games):
        appid_list.append(played_games[i])

    title_list = []
    for ID in appid_list:
        if ID in game_ratings1['appid']:
            game = game_ratings1[game_ratings1['appid'] == ID]
            game_title = game.iloc[0]['title']
            title_list.append(game_title)
    count = 0      
    for rec in rec_list:
        if rec in title_list:
            rec_list.remove(rec)
            count += 1
        elif rec == your_game:
            rec_list.remove(rec)
            count +=1
    rec_list.sort()

    return your_game, rec_list, count

#%%
#Choose to run on a specific user
your_game, rec_list, count = game_recommendation(user_data[100])
print("Your Top Game:", your_game, "\nRecommendations based on Your Top Game:\n", rec_list)
print("Games Omiited: ", count)

#%%
#Run for all users
#Stops on user 80 - not sure why
for user in user_data:
    your_game, rec_list, count = game_recommendation(user)
    print("Your Top Game:", your_game, "\nRecommendations based on Your Top Game:\n", rec_list)
    print("Games Omiited: ", count)
