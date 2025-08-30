import matplotlib.pyplot as plt
import seaborn as sns

def plot_text_length_distribution(df):
    # Create text_length column for plotting
    df_with_text_length = df.copy()
    df_with_text_length['text_length'] = df_with_text_length['video_transcription_text'].str.len()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_with_text_length, x="text_length", hue="claim_status", multiple="dodge", kde=False)
    plt.title("Distribution of Text Length for Claims and Opinions")
    plt.xlabel("Video Transcription Text Length")
    plt.ylabel("Count")
    plt.savefig("text_length_distribution.png")
    plt.show()  # Changed from plt.close() to plt.show() to see the plot
    plt.close()