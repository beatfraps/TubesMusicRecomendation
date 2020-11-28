
from flask import Flask, render_template, request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#create the object of Flask
app  = Flask(__name__)



@app.route('/')
def index():
    popularDataset = popular()
    popularArtist = popularDataset['artist-name'].head(5)
    totalPlay = popularDataset['total_artist_plays'].head(5)

    return render_template('index.html', artists = popularArtist , totalPlays = totalPlay)

@app.route('/login', methods=['POST','GET'])
def login():
    if request.method == 'POST':
        name = request.form.get('name')
        result = rekomendasi(name)

        artist = result[0]
        result.remove(artist)

        return render_template('home.html', artist = artist, name = result)



def popular():
    global query_index
    user_data = pd.read_table('dataset/usersha1-artmbid-artname-plays.tsv',
                              names=['users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                              usecols=['users', 'artist-name', 'plays'])

    user_profiles = pd.read_table('dataset/usersha1-profile.tsv',
                                  names=['users', 'gender', 'age', 'country', 'signup'],
                                  usecols=['users', 'country'])
#mengambil data user berdasarkan artistname dan menjumlahkan play berdasarkan artist name dan mengubah nama playsnya menjadi total artist play
    artist_plays = (user_data.
        groupby(by=['artist-name'])['plays'].
        sum().
        reset_index().
        rename(columns={'plays': 'total_artist_plays'})
    [['artist-name', 'total_artist_plays']]
        )
#menyatukan data user_data dengan artist_plays berdasarkan artistname yang ada pada user_data dan artist_name yang ada oada artist_play
    user_data_with_artist_plays = user_data.merge(artist_plays, left_on='artist-name', right_on='artist-name',
                                                  how='left')
#popularity_threshold berguna untuk mengambil data dengan plays 100000 keatas
    popularity_threshold = 100000
    user_data_popular_artists = user_data_with_artist_plays.query('total_artist_plays >= @popularity_threshold')
#menyatukan data user_data dengna artist play diatas dengan user_profiles berdasarkan users pada user_data_with_artists dengan users yang ada di user_profiles
    combined = user_data_popular_artists.merge(user_profiles, left_on='users', right_on='users', how='left')
#mengambil data berdasarkan query country yang isinya indonesia
    id_data = combined.query('country == \'Indonesia\'')
    return id_data

def rekomendasi(name):
    id_data = popular()
#membuat sebuah matrix dengan baris artist-name, colom = user, dengan value plays dan memfilter setiap data yang kosong mejadi 0
    wide_artist_data = id_data.pivot(index='artist-name', columns='users', values='plays').fillna(0)
#mengambil letak matrix yang bukan 0 (https://www.w3schools.com/python/scipy_sparse_data.asp)
    wide_artist_data_sparse = csr_matrix(wide_artist_data.values)

#untuk mengkasifikasikan kemiripan berdasarkan vector artist dengan metric cosine
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(wide_artist_data_sparse)


#untuk mencari data artist yang mirip dengan nama yang sama
    for i in range(0, wide_artist_data_sparse.shape[0]):
        result = []
        if wide_artist_data.index[i] == name:
            query_index = i

        #mengambil n data terdekat dengan fungsi kneighbors dengan mengambil data yang ada pada wide_artisti_data
            distances, indices = model_knn.kneighbors(wide_artist_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)
        #flatten berfungsi untuk mengubah vector yang tadinya 2 dimensi menjadi 1 dimensi
        #format berfungsi untuk menampilkan kembali sesuai format data

            for i in range(0, len(distances.flatten())):
                if i == 0:
                    result.append(wide_artist_data.index[query_index])
                else:
                    result.append(wide_artist_data.index[indices.flatten()[i]])
            break
        else:
            result = ["Hasil tidak ditemukan"]

    return result



#run flask app
if __name__ == "__main__":
    app.run(debug=True)