# Processamento de Linguagem Natural

A primeira ideia era criar um modelo de NLP para `Responder perguntas e gere textos criativos` com base em `Jurassic-1 Jumbo` da `Hugging Face`. <br />
Mas pelo que vi, ia precisar de investimento monetário que eu não tinha.<br />
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



## Tecnologias

Python3<br />
Anaconda

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

## Datasets

Mental Health FAQ for Chatbot - `https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot`<br />
Question-Answer combination - `https://www.kaggle.com/datasets/veeralakrishna/questionanswer-combination`<br />
Yahoo Answers Topics Dataset - `https://www.kaggle.com/datasets/thedevastator/yahoo-answers-topics-dataset`<br />
CoQA Conversational Question Answering Dataset - `https://www.kaggle.com/datasets/jeromeblanchet/conversational-question-answering-dataset-coqa`<br />

## Autor

Igor do E. S. Balbino