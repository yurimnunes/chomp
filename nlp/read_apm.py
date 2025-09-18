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


# --- Exemplo de Uso ---
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

    parser = APMParser()
    modelo_hs23 = parser.parse(apm_file_content)

    print("--- Resultado do Parser (Corrigido) ---")
    pprint(modelo_hs23)