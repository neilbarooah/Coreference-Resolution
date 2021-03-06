import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F

import gtnlplib.utils as utils
import gtnlplib.coref as coref

class BiLSTMWordEmbedding(nn.Module):
    '''
    In this component, you will use a Bi-Directional LSTM to get initial embeddings.
    The embedding for word i is the i'th hidden state of the LSTM
    after passing the sentence through the LSTM.
    '''

    ## deliverable 4.1
    def __init__(self, word_to_ix, word_embedding_dim, hidden_dim, num_layers, dropout):
        '''
        :param word_to_ix: dict mapping words to unique indices
        :param word_embedding_dim: the dimensionality of the input word embeddings
        :param hidden_dim: the dimensionality of the output embeddings that go to the classifier
        :param num_layers: the number of LSTM layers to use
        :param dropout: amount of dropout in LSTM
        '''
        super(BiLSTMWordEmbedding, self).__init__()
        self.word_to_ix = word_to_ix
        self.num_layers = num_layers
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = False

        self.output_dim = hidden_dim

        # STUDENT
        # Construct the needed components in this order:
        # 1. An embedding lookup table
        # 2. The LSTM
        # Note we want the output dim to be hidden_dim, but since our LSTM
        # is bidirectional, we need to make the output of each direction hidden_dim/2
        # name your embedding member "word_embeddings"
        self.word_embeddings = nn.Embedding(len(self.word_to_ix), self.word_embedding_dim)
        self.lstm = nn.LSTM(input_size=self.word_embedding_dim, hidden_size=int(self.hidden_dim/2), num_layers=self.num_layers, dropout=dropout, bidirectional=True)
        
        
        # END STUDENT

        self.hidden = self.init_hidden()

    ## deliverable 4.1
    def forward(self, document):
        '''
        This function has several parts.
        1. Look up the embeddings for the words in the document.
           These will be the inputs to the LSTM sequence model.
           NOTE: At this step, rather than a list of embeddings, it should be a single tensor.
        2. Now that you have your tensor of embeddings, You can pass it through your LSTM.
        3. Convert the outputs into the correct return type, which is a list of lists of
           embeddings, each of shape (1, hidden_dim)
        NOTE: Make sure you are reassigning self.hidden to the new hidden state!
        :param document: a list of strs, the words of the document
        :returns: a list of embeddings for the document
        '''
        assert self.word_to_ix is not None, "ERROR: Make sure to set word_to_ix on \
                the embedding lookup components"
        # STUDENT
        i = ag.Variable( torch.LongTensor([self.word_to_ix[t] for t in document]))
        inp = self.word_embeddings(i).resize(len(document), 1, self.word_embedding_dim)

        out, new_hidden = self.lstm(inp, self.hidden)
        self.hidden = new_hidden
        return [j for j in out]
        # END STUDENT


    def init_hidden(self):
        '''
        PyTorch wants you to supply the last hidden state at each timestep
        to the LSTM.  You shouldn't need to call this function explicitly
        '''
        if self.use_cuda:
            return (ag.Variable(cuda.FloatTensor(self.num_layers * 2, 1, self.hidden_dim//2).zero_()),
                    ag.Variable(cuda.FloatTensor(self.num_layers * 2, 1, self.hidden_dim//2).zero_()))
        else:
            return (ag.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim//2)),
                    ag.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim//2)))

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()

    def to_cuda(self):
        self.use_cuda = True
        self.cuda()

class AttentionBasedMarkableEmbedding(nn.Module):
    '''
    This class accepts embeddings from the entire document and a target markable.
    Its job is to produce a single embedding for that markable based on a trained attention component.
    '''
    
    ## deliverable 4.2
    def __init__(self, embedding_dim):
        '''
        :param embedding_dim: the embedding of inputs to be received,
            also to be used for the attention vector
        '''
        super(AttentionBasedMarkableEmbedding, self).__init__()
        # STUDENT
        self.embedding_dim = embedding_dim
        self.weights = nn.Linear(self.embedding_dim, 1)
        # END STUDENT
        self.use_cuda = False
        
    ## deliverable 4.2
    def forward(self, embeddings, markable):
        '''
        :param embeddings: all embeddings for words in the document
        :param markable: the markable for which we want a weighted embedding
        :returns: attended embedding for markable (1d vector)
        '''
        start_ind = markable.start_token
        end_ind = markable.end_token
        soft = nn.Softmax(1)
        arr = []
        for i in range(start_ind, end_ind):
            arr.append(self.weights(embeddings[i]))
        arr = torch.squeeze(torch.stack(arr, dim=1), 0).view(1, end_ind-start_ind)
        arr = soft(arr)

        final_arr = []
        for ind, embed in enumerate(embeddings[start_ind:end_ind]):
            final_arr.append(arr[0][ind] * embed)

        final_arr = torch.cat(final_arr)
        return torch.sum(final_arr, dim=0)

        
    def to_cuda(self):
        self.use_cuda = True
        self.cuda()

class SequentialScorer(nn.Module):
    '''
    This class scores coreference between markables based on a concatenated embedding input
    Architecture: input embedding -> Linear layer -> ReLU -> Linear layer -> score
    '''
    
    ## deliverable 4.3
    def __init__(self, mark_embedding_dim, feat_set, feat_emb_dim, hidden_dim):
        '''
        :param mark_embedding_dim: dimension of markable embeddings
        :param feat_set: list of features expected to occur
        :param feat_emb_dim: dimension of boolean feature embeddings
        :param hidden_dim: dimension for intermediate representations: 100 + 100 + 4 * 10
        '''
        super(SequentialScorer, self).__init__()
        
        self.feat_set = feat_set
        
        # STUDENT
        # keep this order to pass tests
        self.feat_off_embs = nn.Embedding(len(feat_set), embedding_dim=feat_emb_dim)
        self.feat_on_embs = nn.Embedding(len(feat_set), embedding_dim=feat_emb_dim)
        self.first_layer = nn.Linear(len(feat_set) * feat_emb_dim + 2 * mark_embedding_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, 1)
        
        # END STUDENT
        self.use_cuda = False
    

    # seq = SequentialScorer(TEST_EMBEDDING_DIM, min_features, 2, COREF_FF_HIDDEN)  # 4, min_features, 2, 5
    # emb5 = ag.Variable(torch.rand(1, TEST_EMBEDDING_DIM))
    # emb0 = ag.Variable(torch.rand(1, TEST_EMBEDDING_DIM))
    # pred = float(seq(emb5, emb0, ['exact-match', 'last-token-match']))  # -0.359851

    ## deliverable 4.3
    def forward(self, emb_i, emb_a, pos_feats):
        '''
        :param emb_i: embedding for current markable
        :param emb_a: embedding for antecedent markable
        :param pos_feats: features with positive value
        :returns: score
        :rtype: 1x1 Variable
        '''
        i = ag.Variable(torch.LongTensor([ind for ind, value in enumerate(self.feat_set) if value in pos_feats]))
        j = ag.Variable(torch.LongTensor([ind for ind, value in enumerate(self.feat_set) if value not in pos_feats]))
        arr = []
        for ind, feat in enumerate(self.feat_set):
            if feat in pos_feats:
                feat_emb = self.feat_on_embs(ag.Variable(torch.LongTensor([ind]))).view(1, -1)
            else:
                feat_emb = self.feat_off_embs(ag.Variable(torch.LongTensor([ind]))).view(1, -1)
            arr.append(feat_emb)
        arr = torch.squeeze(torch.stack(arr, dim=1), 0).view(-1, 1)
        concat = torch.cat((emb_i.view(-1, 1), emb_a.view(-1, 1), arr)).view(1, -1)[0]
        return self.second_layer(F.relu(self.first_layer(concat)))

        
    ## deliverable 4.4
    def score_instance(self, doc_embs, markables, i, feats):
        '''
        A function scoring all coref candidates for a given markable
        Don't forget the new-entity option!
        :param doc_embs: embeddings for markables in the document
        :param markables: list of all markables in the document
        :param i: index of current markable
        :param feats: feature extraction function
        :returns: list of scores for all candidates
        :rtype: torch.FloatTensor of dimensions 1x(i+1)
        '''
        def get_pos_feats(markables, a, i):
            return [k for k,v in feats(markables, a, i).items() if v > 0]
        
        # STUDENT
        arr = []
        for j in range(0, i + 1):
            features = get_pos_feats(markables, i, j)
            score = self.forward(doc_embs[i], doc_embs[j], features)
            arr.append(score) # don't use .data; and use cat()
        return torch.cat((arr)).view(1, -1)

        
    ## deliverable 4.4
    def instance_top_scores(self, doc_embs, markables, i, true_antecedent, feats):
        '''
        Find the top-scoring true and false candidates for i in the markable.
        If no false candidates exist, return (None, None).
        You can probably just copy this over from 'coref_learning.py'
        :param doc_embs: list of embeddings for all words in the document
        :param markables: list of all markables in the document
        :param i: index of current markable
        :param true_antecedent: gold label for markable
        :param feats: feature extraction function
        :returns trues_max: best-scoring true antecedent
        :returns false_max: best-scoring false antecedent
        '''
        scores = self.score_instance(doc_embs, markables, i, feats)
        true_entity = markables[true_antecedent].entity
        true_list = []
        false_list = []
        for idx in range(0, i):
            m = markables[idx]
            if m.entity == true_entity:
                true_list.append(idx)
            else:
                false_list.append(idx)
        
        if len(false_list) == 0:
            return None, None

        if len(true_list) == 0:
            true_list.append(i) # if no proper antecendent exists, i itself becomes an antecedent
        else:
            false_list.append(i)

        true_scores = torch.cat([scores[0][j] for j in true_list])
        false_scores = torch.cat([scores[0][j] for j in false_list])
        max_true_score = torch.max(true_scores)
        max_false_score = torch.max(false_scores)
        
        return max_true_score, max_false_score


    def to_cuda(self):
        self.use_cuda = True
        self.cuda()

def train(doc_lstm_model, attn_model, scoring_model, optimizer, words_set, markable_set, feats, word_limit, epochs=2, margin=1.0, use_cuda=False):
    if not use_cuda:
        _zero = ag.Variable(torch.Tensor([0]))
    else:
        _zero = ag.Variable(torch.cuda.FloatTensor([0]))
        doc_lstm_model.to_cuda()
        attn_model.to_cuda()
        scoring_model.to_cuda()
    for ep in range(epochs):
        tot_loss = 0.0
        instances = 0
        doc_losses = []
        for words, marks in zip(words_set, markable_set):
            words = words[:word_limit]
            marks = [m for m in marks if m.end_token < word_limit]
            optimizer.zero_grad()
            doc_lstm_model.clear_hidden_state()

            if not use_cuda:
                loss = ag.Variable(torch.FloatTensor([0.0]))
            else:
                loss = ag.Variable(torch.cuda.FloatTensor([0.0]))

            base_embs = doc_lstm_model(words)
            att_embs = [attn_model(base_embs, m) for m in marks]
            true_ants = coref.get_true_antecedents(marks)
            for i in range(len(marks)):
                max_t, max_f = scoring_model.instance_top_scores(att_embs, marks, i, true_ants[i], feats)
                if max_t is None: continue

                if not use_cuda:
                    marg = ag.Variable(torch.Tensor([margin])) - max_t + max_f
                else:
                    marg = ag.Variable(torch.cuda.FloatTensor([margin])) - max_t + max_f

                loss += torch.max(torch.cat((_zero, marg)))
            instances += len(marks)
            sc_loss = utils.to_scalar(loss)
            tot_loss += sc_loss
            doc_losses.append(f'{sc_loss / len(marks):.5f}')
            loss.backward()
            optimizer.step()
        print(f'Epoch {ep+1} complete.\nDocument losses = {", ".join(doc_losses)}')
        print(f'Overall loss = {tot_loss / instances:.5f}')
        
def evaluate(doc_lstm_model, attn_model, scoring_model, words_set, markable_set, feats):
    doc_lstm_model.eval()
    attn_model.eval()
    scoring_model.eval()
    emb_dict = {} # for getting around matcher's signature
    for words, marks in zip(words_set, markable_set):
        doc_lstm_model.clear_hidden_state()
        base_embs = doc_lstm_model(words)
        att_embs = [attn_model(base_embs, m) for m in marks]
        emb_dict[marks[0].entity] = att_embs # don't try this at home.
    resolver = make_resolver(feats, emb_dict, scoring_model)
    coref.eval_on_dataset(resolver, markable_set)
    return resolver

# helper
def make_resolver(feats, emb_dict, scoring_model):
    return lambda markables : [utils.argmax(scoring_model.score_instance(emb_dict[markables[0].entity], markables, i, feats))\
                               for i in range(len(markables))]

