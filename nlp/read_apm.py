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
                model_info['best'] = float(best_value_str)
            return
        
        if '=' in line and line.lstrip().startswith('obj'):
            model_info['obj'] = line.split('=', 1)[1].strip()
        else:
            model_info['equations'].append(line)

class APMModel:
    """
    Representa um modelo de otimização APM completo, com dados analisados,
    uma função objetivo executável e uma lista de funções de restrição.
    """
    def __init__(self, apm_content: str):
        """
        Inicializa o modelo, analisando o conteúdo e criando as funções.
        """
        parser = APMParser()
        self.data = parser.parse(apm_content)
        self.objective_function = self._create_objective_function()
        
        # NOVO: Chama o método para criar as funções de restrição
        self.constraints = self._create_constraint_functions()

    def _create_objective_function(self):
        """Cria uma função Python a partir da string da função objetivo."""
        if not self.data['obj']:
            return None
        expression_string = self.data['obj'].replace('^', '**')
        try:
            obj_func = lambda x: eval(expression_string, {"x": x})
            return obj_func
        except Exception as e:
            print(f"Erro ao criar a função objetivo: {e}")
            return None

    def _create_constraint_functions(self):
        """
        Cria uma lista de funções de restrição a partir das equações,
        padronizadas para o formato g(x) <= 0.
        """
        constraint_functions = []
        
        # Regex para encontrar a expressão, o operador (>= ou <=) e o valor
        pattern = re.compile(r'(.*)\s*(>=|<=)\s*(.*)')

        for eq_string in self.data['equations']:
            match = pattern.search(eq_string)
            
            if not match:
                print(f"Aviso: A restrição '{eq_string}' não possui '>=' ou '<=' e será ignorada.")
                continue

            # Separa em lado esquerdo (LHS), operador, e lado direito (RHS)
            lhs = match.group(1).strip()
            op = match.group(2).strip()
            rhs = match.group(3).strip()

            # Constrói a nova expressão no formato g(x) <= 0
            if op == '>=':
                # A >= B  --->  B - A <= 0
                g_x_string = f"({rhs}) - ({lhs})"
            elif op == '<=':
                # A <= B  --->  A - B <= 0
                g_x_string = f"({lhs}) - ({rhs})"
            
            # Converte a sintaxe para Python (^ -> **)
            py_expression = g_x_string.replace('^', '**')
            
            try:
                # IMPORTANTE: Usar 'expr=py_expression' para capturar o valor
                # da string da expressão em cada iteração do loop.
                func = lambda x, expr=py_expression: eval(expr, {"x": x})
                constraint_functions.append(func)
            except Exception as e:
                print(f"Erro ao criar função para a restrição '{eq_string}': {e}")
        
        return constraint_functions

    def __repr__(self):
        """Representação do objeto para facilitar a visualização."""
        return f"<APMModel name='{self.data['name']}'>"

if __name__ == '__main__':
    apm_file_content = """
    Model hs23
      Variables
        x[1] = 3, <=50, >=-50
        x[2] = 1, <=50, >=-50
        obj
      End Variables

      Equations
        x[1] + x[2] >= 1
        x[1]^2 + x[2]^2 >= 1
        9*x[1]^2 + x[2]^2 >= 9
        x[1]^2 - x[2] >= 0
        x[2]^2 - x[1] >= 0    

        ! best known objective = 2
        obj = x[1]^2 + x[2]^2
      End Equations
    End Model
    """

    # 1. Criar a instância do modelo
    model = APMModel(apm_file_content)

    print(f"Modelo carregado: {model}")
    print(f"Encontradas {len(model.constraints)} funções de restrição.")
    print("-" * 40)

    # 2. Ponto de teste
    # O ponto (1, 1) viola a segunda restrição (9*1+1 >= 9 -> 10 >= 9) mas satisfaz a primeira (1+1 >= 1)
    # O ponto (3, 1) satisfaz todas as restrições >=
    ponto_teste = {1: 3.0, 2: 1.0}
    print(f"Avaliando restrições para o ponto x = {ponto_teste}:\n")

    # 3. Iterar e testar cada função de restrição
    # Se g(x) <= 0, a restrição é satisfeita (válida).
    # Se g(x) > 0, a restrição é violada (inválida).
    for i, constraint_func in enumerate(model.constraints):
        original_eq = model.data['equations'][i]
        result = constraint_func(ponto_teste)
        
        status = "Satisfeita (Válida)" if result <= 0 else "Violada (Inválida)"
        
        print(f"Restrição Original: '{original_eq}'")
        print(f"Resultado g(x): {result:.4f}  --> Status: {status}\n")