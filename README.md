# 🌾 Dashboard Soja Brasil - Análise Econômica (2008-2024)

**Status:** ✅ Funcional | 🚧 Em desenvolvimento contínuo

## 🎯 Sobre o Projeto

Dashboard interativo desenvolvido com Streamlit para análise econômica da produção de soja brasileira, utilizando dados oficiais de produtividade, custos e preços.

O projeto integra dados de múltiplas fontes em um banco de dados Supabase, permitindo análise histórica e comparativa da rentabilidade da soja por estado e ao longo do tempo.

## 📊 Fontes de Dados

- **CONAB**: Produção, Produtividade, Área
- **IBGE PAM**: Produção agrícola municipal
- **CEPEA/ESALQ**: Preços históricos
- **Armazenamento**: Supabase (PostgreSQL)

## Dados Utilizados

**Fonte principal:** CONAB - Série Histórica de Safras
- Período: 2010-2023
- Justificativa: Especialização em grãos, dados de custos disponíveis

**Validação:** IBGE PAM
- Correlação com CONAB: r = 0.99
- Diferença média: CONAB de 0,3 a 2,2% superior
- Tendências temporais consistentes entre fontes

## 🛠️ Tecnologias

### Backend & Análise
- **Python 3.11+**
- **Pandas** - Manipulação de dados
- **NumPy** - Operações numéricas
- **Matplotlib & Seaborn** - Visualizações

### Frontend & Dashboard
- **Streamlit** - Interface web interativa
- **Plotly** - Gráficos interativos

### Banco de Dados
- **Supabase** - PostgreSQL hospedado (BaaS)
- **supabase-py** - Cliente Python para Supabase

## 🚀 Instalação e Configuração

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/soja-brasil-dashboard.git
cd soja-brasil-dashboard
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure o Supabase

Crie o arquivo `.streamlit/secrets.toml` com suas credenciais:

```toml
[supabase]
url = "https://SEU-PROJETO.supabase.co"
key = "SUA-CHAVE-PUBLICA-AQUI"
```

📖 **Guia completo**: Veja o arquivo [SUPABASE_SETUP.md](SUPABASE_SETUP.md) para instruções detalhadas.

### 4. Execute o dashboard

```bash
streamlit run app.py
```

O dashboard estará disponível em `http://localhost:8501`

## 🔐 Segurança

- ✅ Credenciais protegidas via `.streamlit/secrets.toml`
- ✅ Arquivo `.gitignore` configurado
- ✅ Separação de configurações no `config.py`
- ⚠️ **NUNCA** commite o arquivo `secrets.toml`

## 📈 Funcionalidades

- ✅ **Métricas principais**: Produtividade, preço, lucro e ROI
- ✅ **Análises temporais**: Evolução de indicadores ao longo do tempo
- ✅ **Comparação regional**: Ranking de estados por rentabilidade
- ✅ **Filtros interativos**: Período e estado personalizados
- ✅ **Download de dados**: Exportação de análises em CSV

## 📁 Estrutura
```
projeto-soja-brasil/
├── data/              # Dados brutos e processados
├── notebooks/         # Análises Jupyter
├── outputs/           # Gráficos e tabelas
└── src/              # Scripts auxiliares
```

## 🚀 Roadmap

- [x] Planejamento e estruturação
- [x] Coleta e limpeza de dados
- [x] Integração com Supabase
- [x] Dashboard Streamlit v1
- [x] Análise exploratória de dados
- [ ] Modelagem estatística avançada
- [ ] Deploy em produção (Streamlit Cloud)
- [ ] Testes automatizados

---

## 📂 Estrutura do Projeto

```
soja-brasil-dashboard/
├── .streamlit/
│   └── secrets.toml          # Credenciais (não versionado)
├── data/                     # Dados brutos e processados
├── notebooks/                # Análises Jupyter
├── app.py                    # Dashboard principal
├── config.py                 # Configurações centralizadas
├── requirements.txt          # Dependências Python
├── .env.example              # Template de variáveis
├── SUPABASE_SETUP.md         # Guia de configuração
└── README.md                 # Este arquivo
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'feat: adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto é de código aberto e está disponível sob a licença MIT.

## 👤 Autor

**Maurício**

- 📧 Email: [ms_sangiogo@hotmail.com]
- 💼 LinkedIn: [seu-linkedin]
- 🐙 GitHub: [@mauriciosangiogo]

---

**Desenvolvido com** ❤️ **e** ☕ **para análise de dados do agronegócio brasileiro**