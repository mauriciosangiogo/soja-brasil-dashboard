"""
Configurações centralizadas do projeto Soja Brasil Dashboard

Este módulo gerencia todas as configurações do aplicativo,
incluindo conexão com Supabase, através de variáveis de ambiente
ou do arquivo secrets.toml do Streamlit.
"""

import os
from typing import Optional
import streamlit as st


class SupabaseConfig:
    """
    Gerencia configurações do Supabase com fallback entre diferentes fontes.

    Ordem de prioridade:
    1. Streamlit secrets (.streamlit/secrets.toml)
    2. Variáveis de ambiente
    3. Valores padrão (apenas para desenvolvimento)
    """

    @staticmethod
    def get_url() -> str:
        """Retorna a URL do Supabase"""
        # Tenta buscar do Streamlit secrets
        if hasattr(st, 'secrets') and 'supabase' in st.secrets:
            return st.secrets['supabase']['url']

        # Fallback para variável de ambiente
        url = os.getenv('SUPABASE_URL')
        if url:
            return url

        # Em desenvolvimento, pode retornar erro ou valor padrão
        raise ValueError(
            "SUPABASE_URL não configurada! "
            "Configure em .streamlit/secrets.toml ou variável de ambiente."
        )

    @staticmethod
    def get_key() -> str:
        """Retorna a chave anon/public do Supabase"""
        # Tenta buscar do Streamlit secrets
        if hasattr(st, 'secrets') and 'supabase' in st.secrets:
            return st.secrets['supabase']['key']

        # Fallback para variável de ambiente
        key = os.getenv('SUPABASE_KEY')
        if key:
            return key

        raise ValueError(
            "SUPABASE_KEY não configurada! "
            "Configure em .streamlit/secrets.toml ou variável de ambiente."
        )

    @staticmethod
    def get_service_role_key() -> Optional[str]:
        """
        Retorna a service role key (opcional, apenas para operações admin)

        ATENÇÃO: A service role key tem permissões totais.
        Use apenas quando absolutamente necessário!
        """
        if hasattr(st, 'secrets') and 'supabase' in st.secrets:
            return st.secrets['supabase'].get('service_role_key')

        return os.getenv('SUPABASE_SERVICE_ROLE_KEY')


class AppConfig:
    """Configurações gerais do aplicativo"""

    # Título e descrição
    PAGE_TITLE = "Soja Brasil - Análise Econômica"
    PAGE_ICON = "🌾"

    # Layout
    LAYOUT = "wide"
    SIDEBAR_STATE = "expanded"

    # Cache
    CACHE_TTL = 3600  # 1 hora em segundos

    # Filtros de dados
    ANO_INICIO = 2008
    ANO_FIM = 2024

    # Metas de performance
    META_ROI = 30.0  # ROI em %
    BASELINE_PRODUTIVIDADE = 50.0  # sc/ha


# Exportar configurações
__all__ = ['SupabaseConfig', 'AppConfig']
