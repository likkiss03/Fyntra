"""
This script performs topic modeling using Latent Dirichlet Allocation (LDA) on a dataset of financial reports to 
analyze and tune the number of topics (K) for optimal model performance. It reads preprocessed text data, splits it into training, 
validation, and test sets based on reporting dates and fraud labels, and evaluates LDA models across a range of topic numbers. 
The evaluation metrics include perplexity (on validation data) and coherence (c_v). 
The script visualizes these metrics to assist in selecting the best number of topics for the LDA model.
Usage:
    - Adjust the data splitting section as needed for different experimental setups.
    - Run the script to generate and save a plot comparing perplexity and coherence for different topic counts.
"""
import pandas as pd
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def evaluate_lda(token_list_train, token_list_val, start=10, stop=150, step=10):
    """
    token_lists: list of lists of tokens (pre-tokenized documents)
    returns: lists of Ks, perplexities, coherences
    """

    dictionary = Dictionary(token_list_train)
    dictionary.filter_extremes(no_below=2, no_above=0.25)

    corpus_train = [dictionary.doc2bow(doc) for doc in token_list_train]
    corpus_val = [dictionary.doc2bow(doc) for doc in token_list_val]
    
    ks   = list(range(start, stop+1, step))
    perps = []
    cohs  = []
    
    for k in tqdm(ks, desc="Evaluating LDA models"):

        lda = LdaModel(
            corpus=corpus_train,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            passes=10,
            alpha='auto',
            eta='auto'
        )

        perp = lda.log_perplexity(corpus_val)
        perps.append(perp)
        
        cm = CoherenceModel(
            model=lda,
            texts=token_list_train,
            dictionary=dictionary,
            coherence='c_v'
        )
        cohs.append(cm.get_coherence())
        
        print(f"K={k:>3}  Perplexity={perp:.2f}  Coherence={cohs[-1]:.4f}")
    
    return ks, perps, cohs

if __name__ == "__main__":
    df = pd.read_csv('path/to/filtered_dataframe.csv')
    df['reporting_date'] = pd.to_datetime(df['reporting_date'])
    
    ##########################################################################
    ######## Uncomment lines below to change the train/val/test split ########
    ##########################################################################

    # train 2003 and test and val sampled from post 2003
    #df_train = df[(df['reporting_date'].dt.year == 2003) & (df['word_count'] > df['word_count'].quantile(0.25))]

    #pos = df[(df['reporting_date'].dt.year>2003) & (df['fraudulent']==1)]
    #neg = df[(df['reporting_date'].dt.year>2003) & (df['fraudulent']==0)]

    #pos_sample = pos.sample(int((2*len(df_train))*df['fraudulent'].mean()+1)*2, random_state=42)
    #neg_sample = neg.sample(((2*len(df_train)) - int((2*len(df_train))*df['fraudulent'].mean()+1)) * 2, random_state=42)
    #bowl = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42)

    #df_val, df_test = train_test_split(bowl, test_size=0.5, stratify=bowl['fraudulent'], random_state=42)

    # train 2003 and test 2011 and val sampled from post 2003 to 2008
    #bowl = df[(df['reporting_date'].dt.year>2003) & (df['reporting_date'].dt.year<=2008)]
    #_, df_val = train_test_split(bowl, test_size=0.2, stratify=bowl['fraudulent'], random_state=42)
    df_test = df[df['reporting_date'].dt.year == 2011]

    # train 2008, test 2011 and val sampled from 2003 to 2007
    #df_train = df_train = df[(df['reporting_date'].dt.year == 2008) & (df['word_count'] > df['word_count'].quantile(0.25))]
    #bowl = df[(df['reporting_date'].dt.year>2003) & (df['reporting_date'].dt.year<=2007)]
    #_, df_val = train_test_split(bowl, test_size=0.2, stratify=bowl['fraudulent'], random_state=42)

    # train and val data between 2003-2008 and test 2011
    bowl = df[(df['reporting_date'].dt.year>2003) & (df['reporting_date'].dt.year<=2008)]
    df_train, df_val = train_test_split(bowl, test_size=0.2, stratify=bowl['fraudulent'], random_state=42)
    df_train = df_train[df_train['word_count'] > df_train['word_count'].quantile(0.25)]
    ##########################################################################

    print(f"Training set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

    print(f"Universal fraudulent ratio: {df['fraudulent'].mean():.2f}")
    print(f"Training set fraudulent ratio: {df_train['fraudulent'].mean():.2f}")
    print(f"Validation set fraudulent ratio: {df_val['fraudulent'].mean():.2f}")
    print(f"Test set fraudulent ratio: {df_test['fraudulent'].mean():.2f}")

    texts_train = df_train['mda'].dropna().tolist()
    texts_val = df_val['mda'].dropna().tolist()

    tokens_train = [simple_preprocess(text) for text in texts_train]
    tokens_val = [simple_preprocess(text) for text in texts_val]
    
    ks, perps, cohs = evaluate_lda(tokens_train, tokens_val, start=10, stop=30, step=1)
    
    fig, ax1 = plt.subplots(figsize=(8,5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Number of Topics (K)')
    ax1.set_ylabel('Perplexity (lower better)', color=color)
    ax1.plot(ks, perps, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
#    ax1.set_ylim(bottom=0)
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Coherence (c_v, higher better)', color=color)
    ax2.plot(ks, cohs, marker='s', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
#    ax2.set_ylim(bottom=0)
    
    plt.title('LDA Model Selection: Perplexity vs. Coherence')
    fig.tight_layout()

    plt.savefig('figure_name.png')
