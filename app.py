# ============================================================================
# DASHBOARD SOJA BRASIL - Análise Econômica (2008-2024)
# Maurício - Portfólio Data Science
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client, Client

# Configuração da página
st.set_page_config(
    page_title="Soja Brasil - Análise Econômica",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS customizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CABEÇALHO
# ============================================================================
st.title("🌾 Análise Econômica da Soja Brasileira")
st.markdown("### Período: 2008-2024 | Dados: CONAB, IBGE, CEPEA")
st.markdown("---")

# ============================================================================
# SIDEBAR - Filtros
# ============================================================================
st.sidebar.title("⚙️ Configurações")
st.sidebar.markdown("---")

# Credenciais Supabase (use st.secrets em produção!)
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://pdqoaihshyrnmigymfnd.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBkcW9haWhzaHlybm1pZ3ltZm5kIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ1MjA5NTIsImV4cCI6MjA4MDA5Njk1Mn0.RpFXusg0fMF8z4LVCKESBvdhjsCFD11mxDxQRmM8mYE")

# ============================================================================
# CACHE: Carregar dados (só roda 1x)
# ============================================================================
@st.cache_data(ttl=3600)  # Cache por 1 hora
def carregar_dados():
    """Carrega e processa dados do Supabase"""
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Carregar dados filtrados (2008-2024)
    df_conab = pd.DataFrame(
        supabase.table('df_conab')
        .select("*")
        .gte('ano', 2008)
        .lte('ano', 2024)
        .execute().data
    )
    
    df_custos = pd.DataFrame(
        supabase.table('df_custos')
        .select("uf, ano, custo_total, custo_fixo, custo_variavel, total_renda_fatores")
        .gte('ano', 2008)
        .lte('ano', 2024)
        .execute().data
    )
    
    df_preco = pd.DataFrame(
        supabase.table('df_preco')
        .select("*")
        .gte('ano', 2008)
        .lte('ano', 2024)
        .execute().data
    )
    
    # Merge
    df = df_conab.merge(df_custos, on=['uf', 'ano'], how='left')
    df = df.merge(df_preco[['ano', 'preco_medio_anual']], on='ano', how='left')
    
    # Criar variáveis derivadas
    df['produtividade_sc_ha'] = df['produtividade_kg_ha'] / 60
    df['faturamento_ha'] = df['preco_medio_anual'] * df['produtividade_sc_ha']
    df['lucro_bruto_ha'] = df['faturamento_ha'] - df['custo_total']
    df['roi_percent'] = (df['lucro_bruto_ha'] / df['custo_total']) * 100
    
    return df

# Carregar dados
with st.spinner('🔄 Carregando dados...'):
    df = carregar_dados()

st.sidebar.success(f"✅ {len(df):,} registros carregados")

# ============================================================================
# FILTROS INTERATIVOS
# ============================================================================
st.sidebar.markdown("### 🔍 Filtros")

# Filtro de anos
anos_disponiveis = sorted(df['ano'].unique())
ano_range = st.sidebar.slider(
    "Período:",
    min_value=int(min(anos_disponiveis)),
    max_value=int(max(anos_disponiveis)),
    value=(int(min(anos_disponiveis)), int(max(anos_disponiveis)))
)

# Filtro de estados
estados_disponiveis = ['Todos'] + sorted(df['uf'].unique())
estado_selecionado = st.sidebar.selectbox(
    "Estado:",
    estados_disponiveis
)

# Aplicar filtros
df_filtrado = df[
    (df['ano'] >= ano_range[0]) & 
    (df['ano'] <= ano_range[1])
].copy()

if estado_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['uf'] == estado_selecionado]

st.sidebar.info(f"📊 {len(df_filtrado):,} registros selecionados")

# ============================================================================
# MÉTRICAS PRINCIPAIS (KPIs)
# ============================================================================
st.markdown("## 📊 Indicadores Principais")

col1, col2, col3, col4 = st.columns(4)

with col1:
    prod_media = df_filtrado['produtividade_sc_ha'].mean()
    st.metric(
        label="🌾 Produtividade Média",
        value=f"{prod_media:.1f} sc/ha",
        delta=f"{prod_media - 50:.1f} vs baseline 50 sc/ha"
    )

with col2:
    preco_medio = df_filtrado['preco_medio_anual'].mean()
    st.metric(
        label="💰 Preço Médio",
        value=f"R$ {preco_medio:.2f}/sc"
    )

with col3:
    lucro_medio = df_filtrado['lucro_bruto_ha'].mean()
    st.metric(
        label="💵 Lucro Médio",
        value=f"R$ {lucro_medio:,.0f}/ha",
        delta="Lucro bruto"
    )

with col4:
    roi_medio = df_filtrado['roi_percent'].mean()
    st.metric(
        label="📈 ROI Médio",
        value=f"{roi_medio:.1f}%",
        delta=f"{roi_medio - 30:.1f}% vs meta 30%"
    )

st.markdown("---")

# ============================================================================
# GRÁFICOS
# ============================================================================

# GRÁFICO 1: Evolução Temporal
st.markdown("## 📈 Evolução Temporal")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    evolucao = df_filtrado.groupby('ano').agg({
        'produtividade_sc_ha': 'mean',
        'preco_medio_anual': 'mean'
    }).reset_index()
    
    ax2 = ax.twinx()
    
    # Produtividade
    ax.plot(evolucao['ano'], evolucao['produtividade_sc_ha'], 
            marker='o', linewidth=2.5, color='#2E7D32', label='Produtividade')
    ax.set_xlabel('Ano', fontweight='bold')
    ax.set_ylabel('Produtividade (sc/ha)', fontweight='bold', color='#2E7D32')
    ax.tick_params(axis='y', labelcolor='#2E7D32')
    
    # Preço
    ax2.plot(evolucao['ano'], evolucao['preco_medio_anual'], 
             marker='s', linewidth=2.5, color='#1976D2', label='Preço')
    ax2.set_ylabel('Preço (R$/sc)', fontweight='bold', color='#1976D2')
    ax2.tick_params(axis='y', labelcolor='#1976D2')
    
    ax.set_title('Produtividade e Preço ao Longo do Tempo', fontweight='bold', pad=20)
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    evolucao_lucro = df_filtrado.groupby('ano')['lucro_bruto_ha'].mean().reset_index()
    
    ax.plot(evolucao_lucro['ano'], evolucao_lucro['lucro_bruto_ha'], 
            marker='D', linewidth=2.5, color='#F57C00')
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.fill_between(evolucao_lucro['ano'], 0, evolucao_lucro['lucro_bruto_ha'], 
                     where=(evolucao_lucro['lucro_bruto_ha'] >= 0), 
                     alpha=0.3, color='green', label='Lucro')
    ax.fill_between(evolucao_lucro['ano'], 0, evolucao_lucro['lucro_bruto_ha'], 
                     where=(evolucao_lucro['lucro_bruto_ha'] < 0), 
                     alpha=0.3, color='red', label='Prejuízo')
    
    ax.set_xlabel('Ano', fontweight='bold')
    ax.set_ylabel('Lucro Bruto (R$/ha)', fontweight='bold')
    ax.set_title('Evolução do Lucro Bruto', fontweight='bold', pad=20)
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)

# GRÁFICO 2: Ranking de Estados
st.markdown("## 🏆 Ranking de Estados por ROI")

fig, ax = plt.subplots(figsize=(14, 6))

ranking = df_filtrado.groupby('uf')['roi_percent'].mean().sort_values(ascending=True)
cores = ['#4CAF50' if x > 30 else '#FF9800' if x > 15 else '#F44336' for x in ranking.values]

bars = ax.barh(ranking.index, ranking.values, color=cores, edgecolor='black', linewidth=1.2)

for bar in bars:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
            f'{width:.1f}%', ha='left', va='center', fontweight='bold')

ax.axvline(30, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Meta: 30%')
ax.set_xlabel('ROI Médio (%)', fontweight='bold')
ax.set_title('ROI Médio por Estado', fontweight='bold', pad=20)
ax.legend()
ax.grid(axis='x', alpha=0.3)

st.pyplot(fig)

# ============================================================================
# TABELA DE DADOS
# ============================================================================
st.markdown("## 📋 Dados Detalhados")

with st.expander("🔽 Ver tabela completa"):
    colunas_mostrar = ['uf', 'ano', 'produtividade_sc_ha', 'preco_medio_anual', 
                       'custo_total', 'faturamento_ha', 'lucro_bruto_ha', 'roi_percent']
    
    st.dataframe(
        df_filtrado[colunas_mostrar].sort_values(['ano', 'uf'], ascending=[False, True]),
        use_container_width=True,
        height=400
    )
    
    # Download
    csv = df_filtrado[colunas_mostrar].to_csv(index=False, sep=';', decimal=',')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"soja_brasil_{ano_range[0]}_{ano_range[1]}.csv",
        mime="text/csv"
    )

# ============================================================================
# RODAPÉ
# ============================================================================
st.markdown("---")
st.markdown("""
**📊 Projeto Portfolio: Análise Econômica da Soja Brasileira**  
**Autor:** Maurício  
**Fontes:** CONAB, IBGE, CEPEA  
**Contato:** [LinkedIn](#) | [GitHub](#) | [Email](#)
""")