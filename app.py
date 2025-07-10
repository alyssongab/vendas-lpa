import os
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, render_template, request, Response, url_for, g
from weasyprint import HTML
from werkzeug.utils import secure_filename
from pathlib import Path

# Configuração inicial
matplotlib.use('Agg')
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join('static', 'images'), exist_ok=True)

# <<< ENRIQUECIMENTO: Informações para o cabeçalho >>>
@app.before_request
def before_request():
    g.professor = "Alysson Gabriel"
    g.disciplina = "Linguagem de Programação Avançada"
    # Adicionamos o ano atual diretamente aqui
    g.current_year = datetime.now().year

# Como não estamos salvando o modelo, vamos treiná-lo toda vez.
# Em uma aplicação real, você treinaria offline e carregaria o .joblib.
def get_trained_model(df):
    model = LinearRegression()
    # <<< ENRIQUECIMENTO DE BACKEND: Usando Mês como feature! >>>
    features = ['IndiceTempo', 'Mês']
    target = 'Vendas'
    
    X = df[features]
    y = df[target]
    
    model.fit(X, y)
    return model


# <<< MUDANÇA: Importando LinearRegression para treinar a cada requisição >>>
from sklearn.linear_model import LinearRegression

def processar_dados_e_gerar_grafico(filepath):
    df = pd.read_csv(filepath)
    df['Data'] = pd.to_datetime(df['Data'])
    df.sort_values('Data', inplace=True) # Garante que os dados estão em ordem
    
    # <<< ENRIQUECIMENTO DE BACKEND: Criando features de tempo >>>
    df['IndiceTempo'] = (df['Data'] - df['Data'].min()).dt.days
    df['Mês'] = df['Data'].dt.month # Feature nova para sazonalidade
    
    # Treina o modelo com os dados do usuário
    model = get_trained_model(df.copy())
    
    # <<< ENRIQUECIMENTO DE BACKEND: Previsão usando as novas features >>>
    features = ['IndiceTempo', 'Mês']
    X_hist = df[features]
    df['Previsao'] = model.predict(X_hist)

    last_date = df['Data'].max()
    last_indice = df['IndiceTempo'].max()
    
    future_dates = pd.to_datetime([last_date + pd.DateOffset(months=i) for i in range(1, 7)])
    
    df_future = pd.DataFrame({'Data': future_dates})
    df_future['IndiceTempo'] = (df_future['Data'] - df['Data'].min()).dt.days
    df_future['Mês'] = df_future['Data'].dt.month
    
    df_future['Previsao'] = model.predict(df_future[features])
    
    # Ajustes no Gráfico para ficar mais bonito
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#212529')
    ax.set_facecolor('#212529')
    
    ax.plot(df['Data'], df['Vendas'], label='Vendas Históricas', marker='o', color='#3498db')
    ax.plot(df['Data'], df['Previsao'], label='Linha de Tendência Sazonal', linestyle='--', color='#e74c3c')
    ax.plot(df_future['Data'], df_future['Previsao'], label='Previsão Futura', marker='x', linestyle='--', color='#2ecc71')
    
    ax.set_title('Análise e Previsão de Vendas', color='white')
    ax.set_xlabel('Data', color='white')
    ax.set_ylabel('Vendas (R$)', color='white')
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend(facecolor='#343a40', edgecolor='none', labelcolor='white')
    fig.tight_layout()
    
    grafico_save_path = os.path.join('static', 'images', 'grafico_previsao.png')
    grafico_url_path = 'images/grafico_previsao.png'
    
    plt.savefig(grafico_save_path, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()

    return df, df_future, grafico_save_path, grafico_url_path

@app.route('/')
def index():
    return render_template('index.html', professor=g.professor, disciplina=g.disciplina)

@app.route('/prever', methods=['POST'])
def prever():
    if 'file' not in request.files: return "Nenhum arquivo enviado", 400
    file = request.files['file']
    if file.filename == '': return "Nenhum arquivo selecionado", 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df, df_future, _, grafico_url_path = processar_dados_e_gerar_grafico(filepath)
        
        return render_template(
            'resultado.html', 
            tabela_historico=df.to_dict(orient='records'), 
            tabela_futuro=df_future.to_dict(orient='records'),
            grafico_url=grafico_url_path,
            nome_arquivo=filename,
            timestamp=datetime.now().timestamp(),
            professor=g.professor,
            disciplina=g.disciplina
        )

    return "Erro inesperado", 500

# A rota /gerar-pdf continua funcionando, mas não vou repeti-la aqui por brevidade.
# Apenas certifique-se de que ela também passe 'professor' e 'disciplina' para o template do PDF se quiser o cabeçalho lá também.
@app.route('/gerar-pdf')
def gerar_pdf():
    filename = request.args.get('arquivo')
    if not filename: return "Nome do arquivo não fornecido.", 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return "Arquivo não encontrado.", 404

    df, df_future, grafico_save_path, _ = processar_dados_e_gerar_grafico(filepath)

    grafico_uri = Path(os.path.abspath(grafico_save_path)).as_uri()

    html_renderizado = render_template(
        'relatorio_pdf.html',
        tabela_historico=df.to_dict(orient='records'),
        tabela_futuro=df_future.to_dict(orient='records'),
        grafico_uri=grafico_uri,
        data_geracao=datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        # Você pode adicionar professor=g.professor, disciplina=g.disciplina aqui também
    )
    
    pdf = HTML(string=html_renderizado).write_pdf()
    
    return Response(pdf, mimetype='application/pdf', headers={'Content-Disposition': 'attachment;filename=relatorio_previsao_sazonal.pdf'})

if __name__ == '__main__':
    app.run(debug=True)