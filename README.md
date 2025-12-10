# 📊 Análise Econométrica da Soja Brasileira (2010-2023)

**Status:** 🚧 Em desenvolvimento

## 🎯 Objetivo

Identificar fatores determinantes da rentabilidade na produção de soja 
brasileira através de análise estatística com dados oficiais.

## 📊 Fontes de Dados

- **CONAB**: Produção, Produtividade, Área - URL: https://www.gov.br/conab/pt-br/atuacao/informacoes-agropecuarias/safras/series-historicas/graos/soja/sojaseriehist.xls/view
- **IBGE PAM**: Produção agrícola municipal
- **CEPEA/ESALQ**: Preços históricos

## Dados Utilizados

**Fonte principal:** CONAB - Série Histórica de Safras
- Período: 2010-2023
- Justificativa: Especialização em grãos, dados de custos disponíveis

**Validação:** IBGE PAM
- Correlação com CONAB: r = 0.99
- Diferença média: CONAB de 0,3 a 2,2% superior
- Tendências temporais consistentes entre fontes

## 🛠️ Tecnologias

Python 3.11 | Pandas | NumPy | SciPy | Statsmodels | Matplotlib | Seaborn

## 📁 Estrutura
```
projeto-soja-brasil/
├── data/              # Dados brutos e processados
├── notebooks/         # Análises Jupyter
├── outputs/           # Gráficos e tabelas
└── src/              # Scripts auxiliares
```

## 🚀 Status do Projeto

- [x] Planejamento
- [x] Coleta de dados
- [ ] Análise exploratória
- [ ] Modelagem estatística
- [ ] Publicação

---

**Autor:** Maurício  
**Início:** 2025
```

---

### **4. Copiar o arquivo de planejamento**

Copie o arquivo `PROJETO_PORTFOLIO_SOJA.md` que criei para dentro da pasta do projeto.

---

## ✅ CHECKLIST DE CRIAÇÃO

**Após criar a estrutura, você deve ter:**
```
C:\Users\ms_sa\Documents\projeto-soja-brasil\
│
├── data\
│   ├── raw\              ✅ (vazio por enquanto)
│   └── processed\        ✅ (vazio por enquanto)
│
├── notebooks\            ✅ (vazio por enquanto)
│
├── outputs\
│   ├── figuras\          ✅ (vazio por enquanto)
│   └── tabelas\          ✅ (vazio por enquanto)
│
├── src\                  ✅ (vazio por enquanto)
│
├── .gitignore            ✅ (criar)
├── README.md             ✅ (criar)
├── requirements.txt      ✅ (criar)
└── PROJETO_PORTFOLIO_SOJA.md  ✅ (copiar)