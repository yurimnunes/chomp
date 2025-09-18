import re
from pprint import pprint

class APMParser:
    """
    Uma classe para analisar (parse) arquivos de otimização no formato .apm
    e extrair informações do modelo em um dicionário estruturado.
    """

    def parse(self, apm_content: str) -> dict:
        """
        Analisa o conteúdo de uma string no formato .apm.

        Args:
            apm_content: Uma string contendo o modelo APM.

        Returns:
            Um dicionário com as informações extraídas do modelo,
            contendo as chaves: 'name', 'variables', 'equations', 'obj', 'best'.
        """
        model_info = {
            'name': None,
            'variables': {},
            'equations': [],
            'obj': None,
            'best': None
        }
        current_section = None

        for line in apm_content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            line_lower = line.lower()

            if line_lower.startswith('model'):
                model_info['name'] = line.split()[1]
            elif line_lower.startswith('variables'):
                current_section = 'variables'
            elif line_lower.startswith('equations'):
                current_section = 'equations'
            elif line_lower.startswith(('end variables', 'end equations', 'end model')):
                current_section = None
            elif current_section == 'variables':
                self._parse_variable_line(line, model_info)
            elif current_section == 'equations':
                self._parse_equation_line(line, model_info)

        return model_info

    def _parse_variable_line(self, line: str, model_info: dict):
        """Método auxiliar para analisar uma linha da seção de variáveis."""
        
        # --- ALTERAÇÃO AQUI ---
        # Ignora a linha de declaração 'obj', pois não é uma variável de decisão.
        if line.lower() == 'obj':
            return
        
        match = re.match(r'(\w+)\[(\d+)\]', line)
        if not match:
            return # Ignora outras linhas malformadas ou sem índice

        base_name = match.group(1)
        index = int(match.group(2))

        initial_value = None
        lower_bound = None
        upper_bound = None
        
        parts = line.split('=', 1)
        if len(parts) > 1:
            value_parts = [p.strip() for p in parts[1].split(',')]
            if value_parts[0]:
                initial_value = float(value_parts[0])
            for part in value_parts[1:]:
                if '<=' in part:
                    upper_bound = float(part.replace('<=', '').strip())
                elif '>=' in part:
                    lower_bound = float(part.replace('>=', '').strip())

        if base_name not in model_info['variables']:
            model_info['variables'][base_name] = {}
        
        model_info['variables'][base_name][index] = [initial_value, upper_bound, lower_bound]

    def _parse_equation_line(self, line: str, model_info: dict):
        """Método auxiliar para analisar uma linha da seção de equações."""
        if line.startswith('!'):
            if 'best known objective' in line.lower():
                best_value_str = line.split('=')[-1].strip()
                try:
                    # VERIFICA SE É UMA FRAÇÃO
                    if '/' in best_value_str:
                        num, den = best_value_str.split('/')
                        model_info['best'] = float(num) / float(den)
                    else:
                        # Se não for, tenta a conversão float normal
                        model_info['best'] = float(best_value_str)
                except (ValueError, ZeroDivisionError) as e:
                    print(f"Aviso: Não foi possível converter o 'best objective' ('{best_value_str}'). Erro: {e}")
                    model_info['best'] = None
            return
        
        if '=' in line and line.lstrip().startswith('obj'):
            model_info['obj'] = line.split('=', 1)[1].strip()
        else:
            model_info['equations'].append(line)

# ... (APMParser e o início da APMModel permanecem os mesmos) ...

class APMModel:
    """
    Representa um modelo de otimização APM completo, com dados analisados,
    uma função objetivo executável e uma lista de funções de restrição.
    """
    def __init__(self, apm_content: str):
        # ... (sem alterações) ...
        parser = APMParser()
        self.data = parser.parse(apm_content)
        self.objective_function = self._create_objective_function()
        self.constraints, self.constraint_map = self._create_constraint_functions()

    def _create_objective_function(self):
        # ... (sem alterações) ...
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
        
        # Regex ATUALIZADO para ser "não-ganancioso" (.+?) e mais robusto
        pattern = re.compile(r'(.+?)\s*(>=|<=|=)\s*(.+)')

        for i, eq_string in enumerate(self.data['equations']):
            # ETAPA 1: Limpar comentários e espaços extras da linha
            cleaned_eq = eq_string.split('!', 1)[0].strip()

            if not cleaned_eq: continue # Pula a linha se ela for só um comentário

            match = pattern.fullmatch(cleaned_eq) # Usar fullmatch para garantir que a linha inteira corresponde
            
            if not match:
                print(f"Aviso: A restrição '{cleaned_eq}' está malformada e será ignorada.")
                continue

            lhs, op, rhs = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            
            g_x_expressions = []
            if op == '>=':
                g_x_expressions.append(f"({rhs}) - ({lhs})")
            elif op == '<=':
                g_x_expressions.append(f"({lhs}) - ({rhs})")
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
        # ... (sem alterações) ...
        return f"<APMModel name='{self.data['name']}'>"

if __name__ == '__main__':
    import sys, os
    # No bloco if __name__ == "__main__":
    apm_filename = os.path.join('Data_Instances', 'hs006.apm')

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
    model = APMModel(apm_file_content)

    print(f"Modelo carregado: {model}")
    # Note que teremos 3 funções de restrição para 2 linhas de equações
    print(f"Encontradas {len(model.constraints)} funções de restrição a partir de {len(model.data['equations'])} equações.")
    print("-" * 50)

    # Pontos de teste
    ponto_1 = {1: 2.0, 2: 2.0} # Satisfaz ambas: 2+2>=3 (4>=3) e 2*2=4
    ponto_2 = {1: 1.0, 2: 4.0} # Satisfaz a igualdade (1*4=4) mas viola a desigualdade (1+4>=3 -> 5>=3, ok) - opa, satisfaz as duas também
    ponto_3 = {1: 1.0, 2: 1.0} # Viola ambas: 1+1>=3 (2>=3) é falso, e 1*1=4 é falso

    pontos_de_teste = {"Ponto Válido (2,2)": ponto_1, "Ponto Válido (1,4)": ponto_2, "Ponto Inválido (1,1)": ponto_3}

    for nome, ponto in pontos_de_teste.items():
        print(f"\n--- Avaliando restrições para o {nome}: x = {ponto} ---")
        
        for i, constraint_func in enumerate(model.constraints):
            # Usamos o mapa para encontrar a equação original
            original_eq_index = model.constraint_map[i]
            original_eq = model.data['equations'][original_eq_index]
            
            result = constraint_func(ponto)
            status = "Satisfeita (<= 0)" if result <= 1e-9 else "Violada (> 0)" # Usando tolerância
            
            print(f"Eq. Original: '{original_eq}'")
            print(f"  -> Resultado da função g_{i+1}(x): {result:.4f}  ({status})")