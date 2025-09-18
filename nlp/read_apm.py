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
    Representa um modelo de otimização APM completo, com dados
    analisados e uma função objetivo executável.
    """
    def __init__(self, apm_content: str):
        """
        Inicializa o modelo, analisando o conteúdo do arquivo .apm
        e criando a função objetivo.

        Args:
            apm_content: Uma string contendo o modelo APM.
        """
        parser = APMParser()
        self.data = parser.parse(apm_content)
        self.objective_function = self._create_objective_function()

    def _create_objective_function(self):
        """
        Cria uma função Python a partir da string da função objetivo.
        
        A função gerada espera um dicionário como entrada para a variável 'x',
        com chaves sendo os índices (ex: x[1], x[2]).
        """
        if not self.data['obj']:
            return None

        # 1. Traduzir a sintaxe de .apm para Python (ex: ^ para **)
        # Adicionamos espaços para garantir que não substituímos algo indesejado
        expression_string = self.data['obj'].replace('^', '**')

        # 2. Criar a função dinamicamente usando lambda e eval
        # A função lambda receberá um dicionário 'x'
        # eval() irá calcular a expressão usando esse dicionário 'x'
        try:
            # O dicionário {"x": x} mapeia a variável 'x' na string para 
            # o argumento 'x' passado para a lambda.
            obj_func = lambda x: eval(expression_string, {"x": x})
            return obj_func
        except Exception as e:
            print(f"Erro ao criar a função objetivo: {e}")
            return None

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

    # 1. Criar uma instância do modelo.
    # A análise e a criação da função acontecem automaticamente.
    model = APMModel(apm_file_content)

    print(f"Modelo carregado: {model}")
    print("-" * 30)

    # 2. Acessar a função objetivo gerada
    obj_func = model.objective_function
    
    if obj_func:
        print("Função objetivo criada com sucesso!")
        
        # 3. Testar a função com alguns valores
        # O formato de entrada deve ser um dicionário onde as chaves são os índices
        # da variável 'x' (1, 2, etc.)
        ponto_teste_1 = {1: 3, 2: 1} # O ponto inicial do arquivo
        resultado_1 = obj_func(ponto_teste_1)
        print(f"Calculando f(x={ponto_teste_1}): {resultado_1}")

        ponto_teste_2 = {1: 1.0, 2: 1.0}
        resultado_2 = obj_func(ponto_teste_2)
        print(f"Calculando f(x={ponto_teste_2}): {resultado_2}") # Deve ser 2.0
        
        # Comparando com o valor ótimo conhecido
        print(f"O valor ótimo conhecido é: {model.data['best']}")

    print("-" * 30)
    print("Dados completos do modelo:")
    pprint(model.data)