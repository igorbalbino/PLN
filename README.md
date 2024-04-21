# Processamento de Linguagem Natural

A primeira ideia era criar um modelo de NLP para `Responder perguntas e gere textos criativos` com base em `Jurassic-1 Jumbo` da `Hugging Face`. <br />
Mas pelo que vi, ia precisar de investimento monetário que eu não tinha. <br />
Deu errado '\m/'.<br />
Comandos:
```
pip install transformers
pip install torch

pip install accelerate -U # modulo está deprecado
pip install transformers[torch] # nao é reconhecido

pip install transformers --upgrade # já estava tudo atualizado
```

<hr />

Como segunda tentativa, encontrei informações na internet sobre a possibilidade de treinar um modelo BERT pré-treinado para responder perguntas de forma gratuita. <br />
Vou tomar como base os tutoriais: `https://youtu.be/L8U-pm-vZ4c?si=ruDaCOEzNSIQQq7Y` e `https://youtu.be/DNRiUGtKlVU?si=g74K56_Ngua6dN4v`. <br />

Ambos usam `transformers` mas um deles usa `pinecone`. Uma plataforma que, a primeira vista, parece gratuita, mas pede criação de conta para ter a chave de acesso da API. Acho que vão pedir dinheiro depois. <br />
Fail '\m/'.<br />

Comandos:
```
pip install -qU torch pinecone-client sentence-transformers
```

<hr />

A segunda tentativa funcionou (diferente da primeira) mas também não ficou da forma que eu quero. Quero uma IA que responda qualquer pergunta, como o ChatGPT ou o Gemini da Google. <br />
Achei um novo tutorial: `https://youtu.be/nscku-jLjSQ?si=Hp8ZbrQDbNpx9Cgy`. <br />
Vou implementar dessa maneira para ver se funciona do jeito que quero agora.

Comandos:
```
pip install gradio git+https://github.com/huggingface/transformers.git
```

Esse funcionou bem!! :D <br />
Tratase de um GPT-2, muito bom, mas não compreende português. Só inglês e da respostas estranhas...

<hr />

Como quarta tentativa, encontrei 2 novos tutoriais: `https://youtu.be/auhqTBpiC6U?si=J4bJXvvNbhCAnk2C` e `https://youtu.be/tEV_Jtmx2cc?si=4kJkQbSmt17PxDau`. <br />
Vou usá-los agora. Junto disso, gerei esse guia no Gemini:
```
Implementing a text-to-text generation model with LSTMs from scratch requires a good understanding of deep learning concepts and coding proficiency. Here's a breakdown of the key steps involved:

1. Data Preprocessing:
Load your text data: This could be from a text file, dataset, or API.
Text cleaning: Remove punctuation, stop words, lowercase everything, and handle other normalization steps depending on your data.
Tokenization: Convert your text into a sequence of tokens (words or sub-words).
Vocabulary building: Create a dictionary mapping each unique token to an integer index.
Sequence creation: Convert each sentence/paragraph into a sequence of integer indices based on the vocabulary.
Padding: Ensure all sequences have the same length by padding shorter sequences with a special token (e.g., 0).

2. Model Architecture:
Embedding Layer: This layer takes integer-encoded sequences and converts them into dense vector representations.
LSTM Layer: The core component of your model. LSTMs process the sequence of embedded vectors, capturing long-range dependencies within the text. You can have multiple LSTM layers stacked for more complex models.
Dense Layer: This layer takes the output from the final LSTM layer and projects it into a dense vector of size equal to your vocabulary size.
Softmax Layer: This layer applies the softmax activation, converting the dense vector into a probability distribution over your vocabulary, where each element represents the probability of the next word in the generated sequence.

3. Training Process:
Define your loss function: Categorical cross-entropy is commonly used for text generation tasks.
Choose an optimizer: Adam is a popular optimizer for neural networks.
Training loop:
Iterate over your training data in batches.
Forward pass: Pass sequences through the model to get predicted probabilities for the next word.
Calculate the loss between predicted probabilities and actual next words.
Backpropagation: Propagate the loss back through the network to update weights and biases.
Update model parameters using the optimizer.
Monitor training progress (loss, accuracy).

4. Text Generation:
Start with a seed sequence: This is the initial text you want the model to build upon.
Encode the seed sequence: Convert the seed sequence into integer indices based on the vocabulary.
Start a loop:
Pass the encoded seed sequence through the model to get the predicted probability distribution for the next word.
Sample a word from the predicted distribution (techniques like random sampling or top-k sampling can be used).
Append the sampled word to the generated text and encode it for the next iteration.
Continue the loop for a predefined length or until a stop token is generated.

Here are some resources that can help you implement this from scratch:
Blog post: https://www.analyticsvidhya.com/blog/2022/03/an-overview-on-long-short-term-memory-lstm/ (This provides a Python implementation with explanations)
YouTube video: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/ (This offers a visual explanation)
Keras documentation: https://keras.io/api/layers/recurrent_layers/lstm/ (For understanding LSTM layers)

Important Considerations:
Implementing an NLP model from scratch requires significant effort and coding expertise.
Pre-built libraries like TensorFlow and PyTorch can significantly simplify the process. Consider using them with pre-trained word embeddings like Word2Vec or GloVe to improve performance.
Training such models can be computationally expensive. Ensure you have adequate hardware resources.
Hyperparameter tuning is crucial for optimal performance. Experiment with different learning rates, batch sizes, LSTM units, etc.

Alternatives:
Explore pre-trained text-to-text generation models like GPT-3 or BART. These offer impressive results but might have limitations on access and usage.
Consider using libraries like Hugging Face Transformers that provide pre-trained models and easy-to-use APIs for text generation.
By understanding the steps involved and utilizing the provided resources, you can embark on the journey of building your own text-to-text generation model with LSTMs. However, if you're starting out, exploring pre-built libraries or APIs might be a more practical approach for faster experimentation and results.
```

Comandos:
```
pip install tensorflow numpy pandas nltk
```

Deu parcialmente certo... Gera respostas disconexas mas acho q é pq a func predict_next_word não funciona corretamente. <br />

<hr />

Como quinta tentativa, usei um dos tutoriais que estão acima... `https://youtu.be/auhqTBpiC6U?si=J4bJXvvNbhCAnk2C`.

Implementei o código e parece funcionar normal no video mas aqui da problema de memória.
Vou dar um jeito de testar num pc melhor depois...

## Tecnologias

Python3 <br />
Anaconda <br />
Transformers <br />
LSTM - Long-Short Term Memory <br />
RNN - Recurrent Neural Network <br />
Kaggle - Datasets <br />

## Auxiliar

Cópia modelo `Jurassic-1 Jumbo` - `https://huggingface.co/facebook/bart-base`<br />

SkLearn (Model Selection - train_test_split) - `https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html`
Hugging Face - `https://huggingface.co/`<br />
Kaggle - `https://www.kaggle.com/`<br />
Google Gemini - `https://gemini.google.com/`<br />

Outros:
`https://huggingface.co/meta-llama/Meta-Llama-3-8B`<br />
`https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct`<br />
`https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1`<br />
`https://huggingface.co/runwayml/stable-diffusion-v1-5`<br />
`https://huggingface.co/Joeythemonster/anything-midjourney-v-4-1`<br />
``<br />
``<br />
``<br />
``<br />
`https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/`<br />

## Datasets

Mental Health FAQ for Chatbot - `https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot`<br />
Question-Answer combination - `https://www.kaggle.com/datasets/veeralakrishna/questionanswer-combination`<br />
Yahoo Answers Topics Dataset - `https://www.kaggle.com/datasets/thedevastator/yahoo-answers-topics-dataset`<br />
CoQA Conversational Question Answering Dataset - `https://www.kaggle.com/datasets/jeromeblanchet/conversational-question-answering-dataset-coqa`<br />

## Autor

Igor do E. S. Balbino