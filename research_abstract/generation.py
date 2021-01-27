import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from copy import deepcopy
from tqdm.notebook import tqdm


import _pickle as pickle

## Some useful functions to ease the processings
def save(file,name, folder = ""):
    if folder != "":
        outfile = open('./'+folder+'/'+name+'.pickle', 'wb')
    else:
        outfile = open(name+'.pickle', 'wb')
    pickle.dump(file, outfile)
    outfile.close
    
def load(name, folder = ""):
    if folder != "":
        outfile = open('./'+folder+'/'+name+'.pickle', 'rb')
    else:
        outfile = open(name+'.pickle', 'rb')
    file = pickle.load(outfile)
    outfile.close
    return file




def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def generate(
        input_ids=None,
        model = None, 
        max_length=64,
        do_sample=False,
        num_beams=5,
        temperature=1.0,
        top_k=50,
        top_p=1,
        repetition_penalty=1,
        bos_token_id=2,
        pad_token_id=0,
        eos_token_ids=3,
        unk_token_ids = 1,
        length_penalty=1,
        vocab_size = 30000,
        remove_unk_tokens = True,
        num_return_sequences = 1
                                ):
        r""" Generates sequences for models with a LM head. The method currently supports greedy or penalized greedy decoding, sampling with top-k or nucleus sampling
        and beam-search.

        Adapted in part from `Facebook's XLM beam search code`_.

        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


        Parameters:

            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.

            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between 1 and infinity. Default to 20.

            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Default to greedy sampling.

            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictely positive. Default to 1.0.

            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

            bos_token_id: (`optional`) int
                Beginning of sentence token if no prompt is provided. Default to 0.

            eos_token_ids: (`optional`) int or list of int
                End of sequence token or list of tokens to stop the generation. Default to 0.
            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.

            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40, bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id)  # do greedy decoding without beam search
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, do_sample=True, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[0][i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id, num_beams=3)  # generate sequences using greedy beam search decoding (3 beams)
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences using using greedy search
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        """

        # We cannot generate if the model does not have a LM head
#        if self.get_output_embeddings() is None:
#            raise AttributeError(
#                "You tried to generate sequences with a model that does not have a LM Head."
#                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
#            )


        max_length = max_length
        do_sample = do_sample
        num_beams = num_beams
        temperature = temperature
        top_k = top_k 
        top_p = top_p 
        repetition_penalty = repetition_penalty
        bos_token_id = bos_token_id
        pad_token_id = pad_token_id
        eos_token_ids = eos_token_ids 
        length_penalty = length_penalty
        
        input_ids = [elt for elt in input_ids for i in range(num_return_sequences)]

        if input_ids is not None:
            batch_size = len(input_ids)  # overriden by the input batch_size
        else:
            batch_size = 1
        if isinstance(eos_token_ids, int):
            eos_token_ids = eos_token_ids

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
        assert temperature > 0, "`temperature` should be strictely positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, "`bos_token_id` should be a positive integer."
        assert isinstance(pad_token_id, int) and pad_token_id >= 0, "`pad_token_id` should be a positive integer."
#        assert isinstance(eos_token_ids, (list, tuple)) and (
#                e >= 0 for e in eos_token_ids
#            ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert length_penalty > 0, "`length_penalty` should be strictely positive."

        if input_ids is None:
            input_ids = np.zeros((batch_size, 1)) + bos_token_id

#        else:
#            assert len(input_ids.shape) == 2, "Input prompt should be of shape (batch_size, sequence length)."

            # current position and vocab size
#        cur_len = input_ids.shape[1]
        vocab_size = model.output.shape[-1]

    #        if num_return_sequences != 1:
    #            # Expand input to num return sequences
    #            input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
    #            input_ids = input_ids.contiguous().view(
    #                batch_size * num_return_sequences, cur_len
    #            )  # (batch_size * num_return_sequences, cur_len)
    #            effective_batch_size = batch_size * num_return_sequences
    #        else:
    #            effective_batch_size = batch_size

        if num_beams > 1:
            output = _generate_beam_search(
                    model,
                    input_ids,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    batch_size,
                    length_penalty,
                    num_beams,
                    vocab_size,
                    unk_token_ids,
                    remove_unk_tokens,
                    bos_token_id  
                )
        else:
            output = _generate_no_beam_search(
                    model,
                    input_ids,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    batch_size,
                    unk_token_ids,
                    remove_unk_tokens,
                    bos_token_id        
                )
        return output



def _generate_no_beam_search(
        model,
        input_ids,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        unk_token_ids = 1,
        remove_unk_tokens = True,
        bos_token_id = 2
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = np.ones(batch_size)
        
        inputs_save = deepcopy(input_ids)
        lengths = np.array([len(elt) for elt in inputs_save])
        input_ids = pad_sequences(input_ids, maxlen=max_length, dtype='int32', padding='post', truncating='post',value=0.0)
        
        
        past = None
        for cur_len in tqdm(range(0, max_length - 1)):
            
            keep_origin = (cur_len+1 >= lengths)*1

            model_inputs = pad_sequences(input_ids, maxlen=max_length, dtype='int32', padding='post', truncating='post',value=0.0)
            
            outputs = model.predict(model_inputs)
            next_token_logits = outputs[:, cur_len, :]

                ## Normalize as probas and remove UNK token prediction
            if remove_unk_tokens:
                next_token_logits[:, unk_token_ids] =  -float("Inf")
                next_token_logits[:, bos_token_id] =  -float("Inf")

                # if model has past, then set the past variable to speed up decoding


                # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in input_ids[i, :cur_len]:
                            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                    # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    # Sample
                next_token = []
                for elt in next_token_logits:
                    elt = softmax(elt)
                    tok = np.random.choice(list(range(elt.shape[0])), p = elt)
                    next_token.append(tok)
                next_token = np.array(next_token)

            else:
                    # Greedy decoding
                next_token = np.argmax(next_token_logits, axis=-1)

                # update generations and finished sentences
                
            tokens_to_add = next_token * unfinished_sents * keep_origin + pad_token_id * (1 - unfinished_sents) + model_inputs[:, cur_len + 1]*(1-keep_origin) 
            
#            input_ids = np.concatenate([input_ids, tokens_to_add.reshape((batch_size, 1))], axis=-1)
            input_ids[:, cur_len + 1] = tokens_to_add
            
            for i, elt in enumerate(tokens_to_add):
                if elt == eos_token_ids:
                    unfinished_sents[i] = 0                           


                # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # add eos_token_ids to unfinished sentences
        if cur_len == max_length:
            input_ids[:, -1] = eos_token_ids

        return input_ids

tokenizer = load('IFart_tokenizer')
    
def _generate_beam_search(
        model,
        input_ids,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
        unk_token_ids = 1,
        remove_unk_tokens = True,
        bos_token_id = 2
        ):
        """ Generate sequences for each example with beam search.
        """
        # Expand input to num beams
        inputs_save = deepcopy(input_ids)
        
        
        beam_scores = [0 if i == 0 else -1e9 for elt in input_ids for i in range(num_beams)]
        input_ids = [elt for elt in input_ids for i in range(num_beams)]
        
#        print(beam_scores)
        
        lengths = np.array([len(elt) for elt in inputs_save])
        input_ids = pad_sequences(input_ids, maxlen=max_length, dtype='int32', padding='post', truncating='post',value=0.0)

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]

        # cache compute states
        past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        for cur_len in tqdm(range(0, max_length - 1)):
            
            keep_origin = (cur_len+1 < lengths)
            
            model_inputs = pad_sequences(input_ids, maxlen=max_length, dtype='int32', padding='post', truncating='post',value=0.0)
            
            outputs = model.predict(model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            scores = outputs[:, cur_len, :]  # (batch_size * num_beams, vocab_size)
                        
            if remove_unk_tokens:
                scores[:, unk_token_ids] =  -float("Inf")
                scores[:, bos_token_id] =  -float("Inf")

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in input_ids[i, :cur_len]:
                            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty
            
            
            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample 2 next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = []
                for elt in scores:
                    elt = softmax(elt)
                    toks = np.random.choice(list(range(elt.shape[0])), p = elt, size = 2, replace = False)
                    next_words.append(toks)
                next_words = np.array(next_words)
#                print(next_words)
                _scores = np.array([np.log(softmax(score)) for score in scores])
                _scores = np.array([[_scores[i,toks[0]], _scores[i,toks[1]]] for i, toks in enumerate(next_words)])
#                print(_scores)
                next_scores = _scores + np.array([beam_scores, beam_scores]).T
                # Match shape of greedy beam search

                next_words = next_words.reshape(batch_size, 2 * num_beams)
                next_scores = next_scores.reshape(batch_size, 2 * num_beams)
                
            else:
                # do greedy beam search
                scores = np.array([np.log(softmax(score)) for score in scores])  # (batch_size * num_beams, vocab_size)
                assert scores.shape == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
#                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                _scores = scores + np.array([beam_scores for elt in range(scores.shape[1])]).T # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
#                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)        
                next_words = np.argsort(_scores, axis = 1)[:, -2:]
                next_scores = np.sort(_scores, axis = 1)[:, -2:]
        
                next_words = next_words.reshape(batch_size, 2*num_beams)
                next_scores = next_scores.reshape(batch_size, 2*num_beams)
                
                
#                next_scores, next_words = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
#            print(next_words)
#            print(next_scores)
            assert next_scores.shape == next_words.shape == (batch_size, 2 * num_beams)
    
            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_ex in range(batch_size):
                # if we are done with this sentence
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []
                print(next_words)
                print(next_scores)
                # next words for this sentence
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):
                    
                    if keep_origin[batch_ex]:
                        idx = inputs_save[batch_ex][cur_len+1]
                        score = 0
                    print(idx)
                    print(score)
                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    if word_id == eos_token_ids or cur_len + 2 == max_length:
                        generated_hyps[batch_ex].add(
                            deepcopy(input_ids[batch_ex * num_beams + beam_id, :cur_len]), score
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_length else num_beams
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            
#            print(beam_scores)
#            print(next_batch_beam)
#            print(input_ids)
#            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
#            beam_words = input_ids.new([x[1] for x in next_batch_beam])
#            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            
            beam_scores = np.array([x[0] for x in next_batch_beam])
            beam_words = np.array([x[1] for x in next_batch_beam])
            beam_idx = np.array([x[2] for x in next_batch_beam])
            
            # re-order batch
            input_ids = input_ids[beam_idx, :]
            
#            print(beam_idx)
#            print(input_ids)
#            print(beam_words)
            
            ins = []
            for i, elt in enumerate(input_ids):
                elt = elt[elt != 0]
                if cur_len > len(elt):
                    ins.append(list(elt) + [beam_words[i]])
                else:
                    ins.append(elt)
            
            input_ids = pad_sequences(ins, maxlen=max_length, dtype='int32', padding='post', truncating='post',value=0.0)
            
            print(tokenizer.sequences_to_texts(input_ids))
#            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)
            
            
            # re-order internal states
#            if past:
#                reordered_past = []
#                for layer_past in past:
#                    # get the correct batch idx from layer past batch dim
#                    # batch dim of `past` and `mems` is at 2nd position
#                    reordered_layer_past = [layer_past[:, i].unsqueeze(1).clone().detach() for i in beam_idx]
#                    reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
#                    # check that shape matches
#                    assert reordered_layer_past.shape == layer_past.shape
#                    reordered_past.append(reordered_layer_past)
#                past = tuple(reordered_past)

            # update current length
#            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(batch_size):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = np.zeros(batch_size)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
#            print(hypotheses.hyp)
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        tgt_len = tgt_len.astype(int)
        decoded = np.zeros((batch_size, tgt_len.max()))
#        print(decoded)
#        print(tgt_len)
#        print(best)
        for i, hypo in enumerate(best):
            decoded[i, : tgt_len[i] - 1] = hypo
            decoded[i, tgt_len[i] - 1] = eos_token_ids

        return decoded


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        for i in range(logits.shape[0]):
            l = logits[i]
            m = np.sort(l)[-max(top_k, min_tokens_to_keep)]
            
            logits[i, logits[i, :] < m] = filter_value
        

    if top_p < 1.0:
        
        
        for i in range(logits.shape[0]):
            l = logits[i]
            l1 = np.sort(softmax(l))
            
            s = 0
            j = 0
            while s < top_p:
                j += 1
                s += l1[-j]
            
            m = np.sort(l)[-max(j,min_tokens_to_keep)]
            
            logits[i, logits[i, :] < m] = filter_value
        
    return logits


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty
