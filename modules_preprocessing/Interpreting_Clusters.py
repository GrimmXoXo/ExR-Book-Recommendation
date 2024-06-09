from itertools import islice
import plotly.graph_objects as go
import pandas as pd

class Interpreting_Genre:
    def __init__(self, n_clusters=0, top_k=15):
        self._n_clusters = n_clusters
        self._top_k = top_k
        self._topk_dict = {}

    def __unique_genres(self, genres_df):
        unique_genres_in_cluster = {}

        for cluster_id in range(self._n_clusters):
            unique_entries = {}  # Using a dictionary to store unique genres and their counts
            
            clustered_genres = genres_df['genres'][genres_df['clusters'] == cluster_id]
            
            for sublist in clustered_genres:
                for genre in sublist:
                    if genre in unique_entries:
                        unique_entries[genre] += 1  # Increment count if genre already exists
                    else:
                        unique_entries[genre] = 1   # Initialize count if genre is new
                
            unique_genres_in_cluster[cluster_id] = unique_entries

        return unique_genres_in_cluster

    @property
    def n_clusters(self):
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, value):
        self._n_clusters = value

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        self._top_k = value

    @property
    def topk_dict(self):
        return self._topk_dict

    @topk_dict.setter
    def topk_dict(self, value):
        self._topk_dict = value

    def take(self, n, iterable):
        """Return the first n items of the iterable as a list."""
        return list(islice(iterable, n))

    def calculate_topk(self, genres_df):
        unique_genres_in_cluster = self.__unique_genres(genres_df)
        topk = {}
        for i in range(self._n_clusters):
            sorted_dict = dict(sorted(unique_genres_in_cluster[i].items(), key=lambda item: item[1], reverse=True))
            topx = self.take(self._top_k, sorted_dict.items())
            topk[i] = topx
        self._topk_dict = topk
        return self._topk_dict

    def plot_graph(self):
        for i in range(self._n_clusters):
            # Convert to DataFrame
            genre_freq_df = pd.DataFrame(self._topk_dict[i], columns=['Genre', 'Frequency'])

            # Create pie chart
            fig = go.Figure(data=[go.Pie(labels=genre_freq_df['Genre'], values=genre_freq_df['Frequency'])])
            
            # Customize layout
            fig.update_layout(
                title_text=f'Genre Frequencies - Cluster {i}',
                showlegend=True,
                legend=dict(x=0.8, y=0.5)
            )

            # Show plot
            fig.show()

