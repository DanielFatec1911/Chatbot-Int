import json
import os
import pika
import time
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS # NOVO: Importação para resolver erros de CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_modelgit init import LogisticRegression
import joblib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # NOVO: Habilita CORS para todas as rotas (necessário para o HTML local)

# --- Configurações de Ambiente ---
# O hostname 'rabbitmq' é resolvido automaticamente pelo Docker Compose
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')
RABBITMQ_QUEUE = 'trade_commands_queue'
# O hostname 'gatewayapi' é resolvido automaticamente pelo Docker Compose
GATEWAY_URL = os.environ.get('GATEWAY_URL', 'http://gatewayapi:8080')

# Variáveis globais para o modelo de PLN
vectorizer = None
model = None
LABELS = [
    "consulta_saldo_btc", 
    "consulta_saldo_eth", 
    "consulta_cotacao", 
    "comando_compra", 
    "comando_venda", 
    "saudacao",
    "despedida",
    "ajuda"
]

# --- Dados de Treinamento (Exemplo) ---
# Em um projeto real, estes dados viriam de um banco de dados ou arquivo externo
TRAINING_DATA = {
    "consulta_saldo_btc": [
        "qual meu saldo de bitcoin", "quanto eu tenho de btc", "saldo btc", "mostrar btc", 
        "ver meu saldo de bitcoin", "btc na carteira"
    ],
    "consulta_saldo_eth": [
        "qual meu saldo de ethereum", "quanto eu tenho de eth", "saldo eth", "mostrar eth",
        "ver meu saldo de ethereum", "eth na carteira"
    ],
    "consulta_cotacao": [
        "qual a cotação do bitcoin", "preço atual do btc", "quanto vale o eth", 
        "valor do ethereum agora", "cotacao btc", "cotacao eth", "preco da cripto"
    ],
    "comando_compra": [
        "quero comprar 1 bitcoin", "comprar 5 eth", "fazer compra de btc", "compra de 10 eth",
        "executar ordem de compra", "comprar bitcoin agora"
    ],
    "comando_venda": [
        "quero vender 2 bitcoin", "vender 3 eth", "fazer venda de btc", "venda de 5 eth",
        "executar ordem de venda", "vender ethereum agora"
    ],
    "saudacao": [
        "olá", "oi", "bom dia", "boa tarde", "boa noite", "e aí", "tudo bem"
    ],
    "despedida": [
        "tchau", "até logo", "adeus", "obrigado e tchau", "falou"
    ],
    "ajuda": [
        "preciso de ajuda", "o que você faz", "quais comandos", "me ajude"
    ]
}

# --- Funções do Modelo PLN ---
def load_or_train_model():
    """Carrega o modelo ou treina um novo se os arquivos não existirem."""
    global vectorizer, model
    
    # Tenta carregar os modelos pré-salvos
    try:
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        model = joblib.load('logistic_model.joblib')
        print("Modelos PLN carregados com sucesso.")
    except Exception as e:
        print(f"Modelos não encontrados ou falha no carregamento ({e}). Treinando novo modelo...")
        
        # Prepara os dados de treino para o SKLearn
        texts = []
        labels = []
        for label, utterances in TRAINING_DATA.items():
            texts.extend(utterances)
            labels.extend([label] * len(utterances))

        # 1. Cria o vetorizador TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)

        # 2. Treina o modelo de Regressão Logística
        model = LogisticRegression(max_iter=1000)
        model.fit(X, labels)

        # Salva os modelos para uso futuro (dentro do contêiner)
        joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
        joblib.dump(model, 'logistic_model.joblib')
        print("Novo modelo PLN treinado e salvo.")

def predict_intent(text):
    """Prevê a intenção de uma frase de entrada."""
    if not vectorizer or not model:
        return "erro" # Modelo não carregado

    text_vectorized = vectorizer.transform([text.lower()])
    intent = model.predict(text_vectorized)[0]
    
    # Obtém a probabilidade (para fins de debug)
    # proba = model.predict_proba(text_vectorized).max()
    
    return intent

# --- Funções de Comunicação com Microserviços ---

def send_trade_command_to_rabbitmq(intent, trade_details):
    """Envia um comando de compra/venda para o RabbitMQ."""
    try:
        # Tenta conectar ao RabbitMQ (com retentativas)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, port=5672, retry_delay=5, connection_attempts=5))
        channel = connection.channel()
        
        # Declara a fila para garantir que ela existe
        channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
        
        message = {
            "intent": intent,
            "user_id": "MOCK_USER_123", # Usar o ID real do usuário autenticado no futuro
            "details": trade_details,
            "timestamp": time.time()
        }
        
        channel.basic_publish(
            exchange='',
            routing_key=RABBITMQ_QUEUE,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
            )
        )
        connection.close()
        return True
    except Exception as e:
        print(f"ERRO ao enviar para RabbitMQ: {e}")
        return False

def get_data_from_gateway(endpoint):
    """Simula uma requisição síncrona para o API Gateway (Você deve substituir o Mock)."""
    # Em um contêiner real, você usaria 'requests.get()' para chamar o GATEWAY_URL.
    
    # --- MOCK SIMPLES para fins de demonstração (Substituir por requests.get() no projeto real) ---
    if "saldo/btc" in endpoint:
        return {"success": True, "value": "0.15 BTC", "currency": "Bitcoin"}
    elif "saldo/eth" in endpoint:
        return {"success": True, "value": "2.4 ETH", "currency": "Ethereum"}
    elif "cotacao/btc" in endpoint or "cotacao/eth" in endpoint:
        if "btc" in endpoint:
            return {"success": True, "price": 60000.00, "currency": "USD"}
        else:
            return {"success": True, "price": 3200.00, "currency": "USD"}
    
    # Se o Gateway real não estiver rodando (nosso placeholder Nginx), isso falharia, 
    # mas a lógica acima garante que a resposta do Chatbot funciona.
    return {"success": False, "message": "Gateway ou endpoint indisponível."}


# --- NOVAS ROTAS PARA SERVIR O FRONTEND HTML ---
@app.route('/')
def serve_frontend():
    """Rota principal que serve o arquivo index.html."""
    try:
        # Lê o conteúdo do index.html que está na mesma pasta do app.py
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Serve o conteúdo do HTML
        return render_template_string(html_content)
    except FileNotFoundError:
        return "Erro: Arquivo index.html não encontrado no contêiner.", 404


# --- Endpoint Principal da API ---

@app.route('/chat', methods=['POST'])
def chat():
    """Recebe a mensagem do usuário, processa e retorna a resposta."""
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"response": "Por favor, envie uma mensagem."}), 400

    # 1. Classificação da Intenção
    intent = predict_intent(user_message)
    response_text = ""

    # 2. Execução da Ação baseada na Intenção
    if intent in ["consulta_saldo_btc", "consulta_saldo_eth"]:
        currency = "btc" if intent == "consulta_saldo_btc" else "eth"
        # Simula chamada síncrona ao Gateway para obter dados
        gateway_response = get_data_from_gateway(f"/api/wallet/saldo/{currency}") 
        
        if gateway_response["success"]:
            response_text = f"Seu saldo de {gateway_response['currency']} é de {gateway_response['value']}."
        else:
            response_text = "Desculpe, não consegui consultar seu saldo. O serviço de carteira está indisponível."

    elif intent == "consulta_cotacao":
        # Extrai a moeda da mensagem (simplificado)
        if "btc" in user_message.lower() or "bitcoin" in user_message.lower():
            currency = "btc"
        elif "eth" in user_message.lower() or "ethereum" in user_message.lower():
            currency = "eth"
        else:
            response_text = "Por favor, especifique a moeda (BTC ou ETH) para a cotação."
            currency = None
            
        if currency:
            gateway_response = get_data_from_gateway(f"/api/coin/cotacao/{currency}")
            if gateway_response["success"]:
                price = gateway_response['price']
                response_text = f"A cotação atual do {currency.upper()} é de ${price:,.2f} USD."
            else:
                response_text = "Não foi possível obter a cotação. O serviço de moedas está fora do ar."

    elif intent in ["comando_compra", "comando_venda"]:
        # Envio de comando para o RabbitMQ
        trade_details = {"amount": 1, "asset": "BTC", "type": intent.split('_')[-1]}
        
        if send_trade_command_to_rabbitmq(intent, trade_details):
            response_text = f"Comando de {trade_details['type']} para {trade_details['amount']} {trade_details['asset']} enviado com sucesso para a fila de processamento (RabbitMQ). Você será notificado sobre a execução."
        else:
            response_text = "Desculpe, o serviço de mensageria (RabbitMQ) está indisponível. Tente novamente mais tarde."
            
    elif intent == "saudacao":
        response_text = "Olá! Como posso ajudar você a gerenciar suas criptomoedas hoje? Pode me perguntar sobre saldo ou cotação."

    elif intent == "despedida":
        response_text = "Até logo! Tenha um ótimo dia."

    elif intent == "ajuda":
        response_text = "Eu posso ajudar você com:\n1. Consultar Saldo (ex: 'qual meu saldo de BTC')\n2. Consultar Cotação (ex: 'preço do ETH')\n3. Enviar Comandos de Trade (ex: 'comprar 1 BTC')"

    else:
        response_text = "Desculpe, não entendi sua intenção. Tente reformular sua pergunta ou peça 'ajuda'."

    # 3. Retorna a resposta para o frontend/Gateway
    return jsonify({"response": response_text, "intent": intent})


# --- Inicialização ---

if __name__ == '__main__':
    # Garante que o modelo é carregado/treinado na inicialização
    load_or_train_model()
    # Inicia a aplicação Flask
    app.run(debug=True, host='0.0.0.0', port=5000)