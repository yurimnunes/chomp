import re
import numpy as np
from pprint import pprint

# add parent directory to path
import os
import sys

# get the parent directory of the current working dir
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import ad as AD

from nlp.nlp import NLPSolver, SQPConfig

# --- Bloco 1: Classes do read_apm.py ---
# Colando as classes que desenvolvemos aqui para criar um arquivo único e funcional.

class APMParser:
    """
    Analisa arquivos de otimização no formato .apm e extrai as informações
    do modelo em um dicionário estruturado.
    """
    def parse(self, apm_content: str) -> dict:
        model_info = {'name': None, 'variables': {}, 'equations': [], 'obj': None, 'best': None}
        current_section = None
        for line in apm_content.strip().split('\n'):
            line = line.strip()
            if not line: continue
            line_lower = line.lower()
            if line_lower.startswith('model'): model_info['name'] = line.split()[1]
            elif line_lower.startswith('variables'): current_section = 'variables'
            elif line_lower.startswith('equations'): current_section = 'equations'
            elif line_lower.startswith(('end variables', 'end equations', 'end model')): current_section = None
            elif current_section == 'variables': self._parse_variable_line(line, model_info)
            elif current_section == 'equations': self._parse_equation_line(line, model_info)
        return model_info

    def _parse_variable_line(self, line: str, model_info: dict):
        if line.lower() == 'obj': return
        match = re.match(r'(\w+)\[(\d+)\]', line)
        if not match: return
        base_name, index = match.group(1), int(match.group(2))
        initial_value, lower_bound, upper_bound = None, None, None
        parts = line.split('=', 1)
        if len(parts) > 1:
            value_parts = [p.strip() for p in parts[1].split(',')]
            if value_parts[0]: initial_value = float(value_parts[0])
            for part in value_parts[1:]:
                if '<=' in part: upper_bound = float(part.replace('<=', '').strip())
                elif '>=' in part: lower_bound = float(part.replace('>=', '').strip())
        if base_name not in model_info['variables']: model_info['variables'][base_name] = {}
        model_info['variables'][base_name][index] = [initial_value, upper_bound, lower_bound]

    def _parse_equation_line(self, line: str, model_info: dict):
        if line.startswith('!'):
            if 'best known objective' in line.lower(): model_info['best'] = float(line.split('=')[-1].strip())
            return
        if '=' in line and line.lstrip().startswith('obj'):
             model_info['obj'] = line.split('=', 1)[1].strip()
        else:
             model_info['equations'].append(line)


class APMModel:
    """
    Representa um modelo APM completo, com dados analisados e funções executáveis.
    """
    def __init__(self, apm_content: str):
        parser = APMParser()
        self.data = parser.parse(apm_content)
        self.objective_function = self._create_objective_function()
        self.constraints, self.constraint_map = self._create_constraint_functions()

    def _create_objective_function(self):
        if not self.data['obj']: return None
        expression_string = self.data['obj'].replace('^', '**')
        try:
            return lambda x: eval(expression_string, {"x": x})
        except Exception as e:
            print(f"Erro ao criar a função objetivo: {e}")
            return None

    def _create_constraint_functions(self):
        constraint_functions = []
        constraint_map = []
        pattern = re.compile(r'(.+?)\s*(>=|<=|=)\s*(.+)')
        for i, eq_string in enumerate(self.data['equations']):
            cleaned_eq = eq_string.split('!', 1)[0].strip()
            if not cleaned_eq: continue
            match = pattern.fullmatch(cleaned_eq)
            if not match:
                print(f"Aviso: A restrição '{cleaned_eq}' está malformada e será ignorada.")
                continue
            lhs, op, rhs = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            g_x_expressions = []
            if op == '>=': g_x_expressions.append(f"({rhs}) - ({lhs})")
            elif op == '<=': g_x_expressions.append(f"({lhs}) - ({rhs})")
            elif op == '=':
                g_x_expressions.append(f"({lhs}) - ({rhs})")
                g_x_expressions.append(f"({rhs}) - ({lhs})")
            for expr in g_x_expressions:
                py_expression = expr.replace('^', '**')
                try:
                    func = lambda x, captured_expr=py_expression: eval(captured_expr, {"x": x})
                    constraint_functions.append(func)
                    constraint_map.append(i)
                except Exception as e:
                    print(f"Erro ao criar função para a restrição '{cleaned_eq}': {e}")
        return constraint_functions, constraint_map

    def __repr__(self):
        return f"<APMModel name='{self.data['name']}'>"


# --- Bloco 2: Funções "Wrapper" para compatibilidade ---

def create_objective_wrapper(apm_objective_func):
    """
    Cria uma função wrapper que aceita um np.ndarray e a converte
    para o formato de dicionário esperado pela função do APMModel.
    """
    def wrapped_f(x_np: np.ndarray) -> float:
        # Converte o array numpy (base 0) para dicionário (base 1)
        x_dict = {i + 1: val for i, val in enumerate(x_np)}
        return apm_objective_func(x_dict)
    return wrapped_f

def create_constraints_wrappers(apm_constraint_funcs: list):
    """
    Cria uma lista de funções de restrição wrapper.
    """
    wrapped_constraints = []
    for apm_c_func in apm_constraint_funcs:
        # Usamos o truque de capturar a função para evitar problemas com escopo em loops
        def wrapped_c(x_np: np.ndarray, func=apm_c_func) -> float:
            x_dict = {i + 1: val for i, val in enumerate(x_np)}
            return func(x_dict)
        wrapped_constraints.append(wrapped_c)
    return wrapped_constraints


# --- Bloco 3: Estrutura do Solver (adaptado de branin.py) ---
# Simulação das classes do solver para que o código seja executável
try:
    from nlp.nlp import NLPSolver, SQPConfig
except ImportError:
    print("Aviso: 'nlp.nlp' não encontrado. Usando classes de simulação (mock).")
    SQPConfig = dict
    class NLPSolver:
        def __init__(self, f, c_ineq, c_eq, x0, config):
            print("Mock NLPSolver inicializado.")
            self.f, self.c_ineq, self.x0 = f, c_ineq, x0
        def solve(self, **kwargs):
            print("--> Mock NLPSolver.solve() chamado")
            # Simula uma solução indo em direção a [1,1]
            x_star = self.x0 * 0.5 + np.array([1,1]) * 0.5
            info = type("Info", (), {"status": "Mock Success"})()
            return x_star, info


def run_solve(name: str, f, c_ineq: list, c_eq: list, x0: np.ndarray,
              mode: str = "auto", max_iter: int = 150):
    """
    Função genérica para configurar e executar o NLPSolver.
    """
    print("=" * 80)
    print(f"Resolvendo problema: '{name}' | modo={mode} | x0={x0}")
    cfg = SQPConfig()
    
    # Configurações podem ser adicionadas aqui se necessário
    # cfg.tol_stat = 1e-5 ...

    solver = NLPSolver(f=f, c_ineq=c_ineq, c_eq=c_eq, x0=x0, config=cfg)
    x_star, info = solver.solve(max_iter=max_iter, tol=1e-8, verbose=True)

    f_star = f(x_star)
    print(f"-> '{name}' CONCLUÍDO. x* = {x_star}, f* = {f_star:.9f}, status = {info.status}")
    print("-" * 80)
    return x_star, f_star, info


# --- Bloco 4: Execução Principal ---

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    import sys, os
    # No bloco if __name__ == "__main__":
    apm_filename = os.path.join('nlp','Data_Instances', 'hs006.apm')

    # 2. Ler o conteúdo do arquivo
    try:
        with open(apm_filename, 'r', encoding='utf-8') as f:
            apm_file_content = f.read()
        print(f"Arquivo '{apm_filename}' lido com sucesso.")
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{apm_filename}' não foi encontrado.")
        print("Por favor, crie o arquivo no mesmo diretório do script.")
        sys.exit(1) # Encerra o script se o arquivo não existir

    print(apm_file_content)
    # 2. Processar a string APM com nosso modelo
    print("--- Processando modelo APM ---")
    apm_model = APMModel(apm_file_content)
    pprint(apm_model.data)

    # 3. Extrair e adaptar as funções e o chute inicial
    print("\n--- Adaptando funções para o formato do solver (numpy) ---")
    
    # Cria a função objetivo compatível com numpy
    f_wrapped = create_objective_wrapper(apm_model.objective_function)

    # Cria a lista de restrições de desigualdade compatíveis com numpy
    c_ineq_wrapped = create_constraints_wrappers(apm_model.constraints)

    # Extrai o x0 e converte para numpy array
    # Assumindo uma única variável base 'x'. Uma lógica mais complexa seria necessária para múltiplas (x, y, z...).
    vars_dict = apm_model.data['variables']['x']
    x0_list = [vars_dict[i][0] for i in sorted(vars_dict.keys())]
    x0_np = np.array(x0_list, dtype=float)

    print(f"Função objetivo e {len(c_ineq_wrapped)} restrições adaptadas.")
    print(f"Vetor de chute inicial: {x0_np}")

    # 4. Chamar o solver com as funções e dados adaptados
    run_solve(name=apm_model.data['name'],
              f=f_wrapped,
              c_ineq=c_ineq_wrapped,
              c_eq=[],
              x0=x0_np,
              mode="ip")