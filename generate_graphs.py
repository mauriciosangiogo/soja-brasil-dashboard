"""
Script para gerar os gráficos do Dashboard Soja Brasil
Salva os gráficos como imagens PNG para visualização
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client
from config import SupabaseConfig, AppConfig
import os

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Criar diretório para salvar os gráficos
os.makedirs('outputs', exist_ok=True)

print("🔄 Conectando ao Supabase...")
supabase = create_client(SupabaseConfig.get_url(), SupabaseConfig.get_key())

print("📊 Carregando dados...")
# Carregar dados
df_conab = pd.DataFrame(
    supabase.table('df_conab')
    .select("*")
    .gte('ano', AppConfig.ANO_INICIO)
    .lte('ano', AppConfig.ANO_FIM)
    .execute().data
)

df_custos = pd.DataFrame(
    supabase.table('df_custos')
    .select("uf, ano, custo_total, custo_fixo, custo_variavel, total_renda_fatores")
    .gte('ano', AppConfig.ANO_INICIO)
    .lte('ano', AppConfig.ANO_FIM)
    .execute().data
)

df_preco = pd.DataFrame(
    supabase.table('df_preco')
    .select("*")
    .gte('ano', AppConfig.ANO_INICIO)
    .lte('ano', AppConfig.ANO_FIM)
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

print(f"✅ {len(df)} registros carregados!")
print(f"📅 Período: {df['ano'].min()} - {df['ano'].max()}")
print(f"🗺️  Estados: {df['uf'].nunique()}")

# GRÁFICO 1: Evolução de Produtividade e Preço
print("\n📈 Gerando Gráfico 1: Evolução Temporal...")
fig, ax = plt.subplots(figsize=(14, 8))

evolucao = df.groupby('ano').agg({
    'produtividade_sc_ha': 'mean',
    'preco_medio_anual': 'mean'
}).reset_index()

ax2 = ax.twinx()

# Produtividade
ax.plot(evolucao['ano'], evolucao['produtividade_sc_ha'],
        marker='o', linewidth=3, color='#2E7D32', label='Produtividade', markersize=8)
ax.set_xlabel('Ano', fontweight='bold', fontsize=12)
ax.set_ylabel('Produtividade (sc/ha)', fontweight='bold', color='#2E7D32', fontsize=12)
ax.tick_params(axis='y', labelcolor='#2E7D32')

# Preço
ax2.plot(evolucao['ano'], evolucao['preco_medio_anual'],
         marker='s', linewidth=3, color='#1976D2', label='Preço', markersize=8)
ax2.set_ylabel('Preço (R$/sc)', fontweight='bold', color='#1976D2', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#1976D2')

ax.set_title('Evolução da Produtividade e Preço da Soja (2008-2024)',
             fontweight='bold', pad=20, fontsize=16)
ax.grid(alpha=0.3)
ax.legend(loc='upper left', fontsize=11)
ax2.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/grafico_1_evolucao_temporal.png', dpi=300, bbox_inches='tight')
print("✅ Salvo: outputs/grafico_1_evolucao_temporal.png")
plt.close()

# GRÁFICO 2: Evolução do Lucro
print("📈 Gerando Gráfico 2: Evolução do Lucro...")
fig, ax = plt.subplots(figsize=(14, 8))

evolucao_lucro = df.groupby('ano')['lucro_bruto_ha'].mean().reset_index()

ax.plot(evolucao_lucro['ano'], evolucao_lucro['lucro_bruto_ha'],
        marker='D', linewidth=3, color='#F57C00', markersize=8)
ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.fill_between(evolucao_lucro['ano'], 0, evolucao_lucro['lucro_bruto_ha'],
                 where=(evolucao_lucro['lucro_bruto_ha'] >= 0),
                 alpha=0.3, color='green', label='Lucro')
ax.fill_between(evolucao_lucro['ano'], 0, evolucao_lucro['lucro_bruto_ha'],
                 where=(evolucao_lucro['lucro_bruto_ha'] < 0),
                 alpha=0.3, color='red', label='Prejuízo')

ax.set_xlabel('Ano', fontweight='bold', fontsize=12)
ax.set_ylabel('Lucro Bruto (R$/ha)', fontweight='bold', fontsize=12)
ax.set_title('Evolução do Lucro Bruto por Hectare (2008-2024)',
             fontweight='bold', pad=20, fontsize=16)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Adicionar valores nos pontos
for i, row in evolucao_lucro.iterrows():
    ax.text(row['ano'], row['lucro_bruto_ha'],
            f"R$ {row['lucro_bruto_ha']:,.0f}",
            ha='center', va='bottom' if row['lucro_bruto_ha'] > 0 else 'top',
            fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/grafico_2_evolucao_lucro.png', dpi=300, bbox_inches='tight')
print("✅ Salvo: outputs/grafico_2_evolucao_lucro.png")
plt.close()

# GRÁFICO 3: Ranking de Estados por ROI
print("📈 Gerando Gráfico 3: Ranking de Estados...")
fig, ax = plt.subplots(figsize=(14, 10))

ranking = df.groupby('uf')['roi_percent'].mean().sort_values(ascending=True)
cores = ['#4CAF50' if x > AppConfig.META_ROI else '#FF9800' if x > 15 else '#F44336'
         for x in ranking.values]

bars = ax.barh(ranking.index, ranking.values, color=cores, edgecolor='black', linewidth=1.5)

for bar in bars:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=10)

ax.axvline(AppConfig.META_ROI, color='black', linestyle='--', linewidth=2.5,
           alpha=0.7, label=f'Meta: {AppConfig.META_ROI}%')
ax.set_xlabel('ROI Médio (%)', fontweight='bold', fontsize=12)
ax.set_title('ROI Médio por Estado (2008-2024)', fontweight='bold', pad=20, fontsize=16)
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/grafico_3_ranking_roi.png', dpi=300, bbox_inches='tight')
print("✅ Salvo: outputs/grafico_3_ranking_roi.png")
plt.close()

# GRÁFICO 4: Matriz de Correlação
print("📈 Gerando Gráfico 4: Matriz de Correlação...")
fig, ax = plt.subplots(figsize=(12, 10))

# Selecionar variáveis numéricas relevantes
vars_correlacao = ['produtividade_sc_ha', 'preco_medio_anual', 'custo_total',
                   'custo_fixo', 'custo_variavel', 'faturamento_ha',
                   'lucro_bruto_ha', 'roi_percent']
df_corr = df[vars_correlacao].corr()

# Gerar heatmap
sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax, vmin=-1, vmax=1)

ax.set_title('Matriz de Correlação - Variáveis Econômicas',
             fontweight='bold', pad=20, fontsize=16)

# Melhorar labels
labels = ['Produtividade', 'Preço', 'Custo Total', 'Custo Fixo',
          'Custo Variável', 'Faturamento', 'Lucro', 'ROI']
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels, rotation=0)

plt.tight_layout()
plt.savefig('outputs/grafico_4_matriz_correlacao.png', dpi=300, bbox_inches='tight')
print("✅ Salvo: outputs/grafico_4_matriz_correlacao.png")
plt.close()

# ESTATÍSTICAS RESUMO
print("\n" + "="*70)
print("📊 ESTATÍSTICAS RESUMO")
print("="*70)

print(f"\n🌾 PRODUTIVIDADE:")
print(f"   • Média: {df['produtividade_sc_ha'].mean():.2f} sc/ha")
print(f"   • Mínima: {df['produtividade_sc_ha'].min():.2f} sc/ha")
print(f"   • Máxima: {df['produtividade_sc_ha'].max():.2f} sc/ha")

print(f"\n💰 PREÇO:")
print(f"   • Média: R$ {df['preco_medio_anual'].mean():.2f}/sc")
print(f"   • Mínimo: R$ {df['preco_medio_anual'].min():.2f}/sc")
print(f"   • Máximo: R$ {df['preco_medio_anual'].max():.2f}/sc")

print(f"\n💵 LUCRO:")
print(f"   • Média: R$ {df['lucro_bruto_ha'].mean():,.2f}/ha")
print(f"   • Mínimo: R$ {df['lucro_bruto_ha'].min():,.2f}/ha")
print(f"   • Máximo: R$ {df['lucro_bruto_ha'].max():,.2f}/ha")

print(f"\n📈 ROI:")
print(f"   • Média: {df['roi_percent'].mean():.2f}%")
print(f"   • Mínimo: {df['roi_percent'].min():.2f}%")
print(f"   • Máximo: {df['roi_percent'].max():.2f}%")

print(f"\n🏆 TOP 5 ESTADOS POR ROI:")
top5 = df.groupby('uf')['roi_percent'].mean().sort_values(ascending=False).head(5)
for i, (estado, roi) in enumerate(top5.items(), 1):
    print(f"   {i}. {estado}: {roi:.2f}%")

print("\n✅ Todos os gráficos foram gerados com sucesso!")
print("📁 Salvos em: outputs/")
