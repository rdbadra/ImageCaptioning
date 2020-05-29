from torchtext.data.metrics import bleu_score
from torchvision.datasets import CocoCaptions

def calculate_bleu(decoder, features, references_corpus, data_loader):
    sampled_ids = decoder.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = data_loader.dataset.id_to_word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    return bleu_score(sentence, references_corpus)


def calc_bleu():
    image_dir = "/home/roberto/Documentos/TFM-UOC/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/"

    json =  image_dir + "annotations/captions_train2014.json"
    cap = CocoCaptions(root = image_dir+"train2014/",
                        annFile = json)

    print('Number of samples: ', len(cap))
    img, target = cap[0] # load 4th sample
    print(target)

# calc_bleu()