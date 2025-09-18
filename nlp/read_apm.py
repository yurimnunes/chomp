import re
from typing import Dict, List, Any, Tuple
import numpy as np

class APMParser:
    def __init__(self):
        self.model_name = ""
        self.variables = []
        self.objective = ""
        self.constraints = []
        self.bounds = {}
        
    def parse(self, apm_content: str) -> Dict[str, Any]:
        """
        Parse APM content and return problem definition
        
        Args:
            apm_content: String content of APM file
            
        Returns:
            Dictionary with problem definition compatible with your professor's solver
        """
        lines = apm_content.split('\n')
        in_variables = False
        in_equations = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('!'):
                continue
                
            # Check for model name
            if line.lower().startswith('model'):
                self.model_name = line.split()[1]
                continue
                
            # Check section boundaries
            if line.lower().startswith('variables'):
                in_variables = True
                in_equations = False
                continue
            elif line.lower().startswith('equations'):
                in_variables = False
                in_equations = True
                continue
            elif line.lower().startswith('end variables'):
                in_variables = False
                continue
            elif line.lower().startswith('end equations'):
                in_equations = False
                continue
            elif line.lower().startswith('end model'):
                break
                
            # Process content based on section
            if in_variables:
                self._parse_variable_line(line)
            elif in_equations:
                self._parse_equation_line(line)
                
        return self._build_problem()
    
    def _parse_variable_line(self, line: str):
        """
        Parse a variable definition line
        Examples:
          x[1] = 1.125, >= 1
          x[2] = 0.125, >= 0
          obj
        """
        # Check if it's just a declaration without value
        if '=' not in line:
            var_name = line.strip()
            self.variables.append({'name': var_name, 'value': 0.0, 'bounds': (None, None)})
            return
            
        # Split variable name and the rest
        parts = line.split('=', 1)
        var_name = parts[0].strip()
        
        # Extract value and constraints
        value_part = parts[1].strip()
        value = 0.0
        bounds = (None, None)
        
        # Extract numerical value if present
        value_match = re.search(r'[-+]?\d*\.?\d+', value_part)
        if value_match:
            value = float(value_match.group())
            
        # Extract bounds
        if '>=' in value_part:
            match = re.search(r'>=\s*([-+]?\d*\.?\d+)', value_part)
            if match:
                bounds = (float(match.group(1)), bounds[1])
        if '<=' in value_part:
            match = re.search(r'<=\s*([-+]?\d*\.?\d+)', value_part)
            if match:
                bounds = (bounds[0], float(match.group(1)))
        if '>' in value_part and '>=' not in value_part:
            match = re.search(r'>\s*([-+]?\d*\.?\d+)', value_part)
            if match:
                bounds = (float(match.group(1)) + 1e-10, bounds[1])  # Add small epsilon for strict inequality
        if '<' in value_part and '<=' not in value_part:
            match = re.search(r'<\s*([-+]?\d*\.?\d+)', value_part)
            if match:
                bounds = (bounds[0], float(match.group(1)) - 1e-10)  # Subtract small epsilon for strict inequality
                
        self.variables.append({'name': var_name, 'value': value, 'bounds': bounds})
        
    def _parse_equation_line(self, line: str):
        """
        Parse an equation line
        Examples:
          obj = (x[1]+1)^3/3 + x[2]
          x[1] + x[2]^2 >= 0
        """
        # Remove comments
        line = re.sub(r'!.*', '', line).strip()
        if not line:
            return
            
        # Check if it's the objective function
        if line.startswith('obj ='):
            self.objective = line.split('=', 1)[1].strip()
        else:
            # It's a constraint
            self.constraints.append(line)
            
    def _build_problem(self) -> Dict[str, Any]:
        """
        Build problem definition compatible with your professor's solver
        """
        # Filter out the objective variable from the variables list
        decision_vars = [var for var in self.variables if var['name'] != 'obj']
        
        # Extract initial values and bounds
        x0 = np.array([var['value'] for var in decision_vars])
        bounds = [var['bounds'] for var in decision_vars]
        
        # Create variable mapping for expression parsing
        var_map = {}
        for i, var in enumerate(decision_vars):
            var_map[var['name']] = i
            
        # Create objective function
        def objective_func(x):
            expr = self.objective
            for var_name, idx in var_map.items():
                expr = expr.replace(var_name, f'x[{idx}]')
            # Replace APM operators with Python operators
            expr = expr.replace('^', '**')
            return eval(expr)
            
        # Create constraint functions
        inequality_constraints = []
        equality_constraints = []
        
        for constraint in self.constraints:
            if '<=' in constraint:
                parts = constraint.split('<=')
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                
                def ineq_constr(x, l=lhs, r=rhs):
                    # Replace variables in expression
                    expr = l
                    for var_name, idx in var_map.items():
                        expr = expr.replace(var_name, f'x[{idx}]')
                    expr = expr.replace('^', '**')
                    lhs_val = eval(expr)
                    
                    # Evaluate right hand side
                    expr = r
                    for var_name, idx in var_map.items():
                        expr = expr.replace(var_name, f'x[{idx}]')
                    expr = expr.replace('^', '**')
                    rhs_val = eval(expr)
                    
                    return lhs_val - rhs_val  # Convert to g(x) <= 0 form
                    
                inequality_constraints.append(ineq_constr)
                
            elif '>=' in constraint:
                parts = constraint.split('>=')
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                
                def ineq_constr(x, l=lhs, r=rhs):
                    # Replace variables in expression
                    expr = l
                    for var_name, idx in var_map.items():
                        expr = expr.replace(var_name, f'x[{idx}]')
                    expr = expr.replace('^', '**')
                    lhs_val = eval(expr)
                    
                    # Evaluate right hand side
                    expr = r
                    for var_name, idx in var_map.items():
                        expr = expr.replace(var_name, f'x[{idx}]')
                    expr = expr.replace('^', '**')
                    rhs_val = eval(expr)
                    
                    return rhs_val - lhs_val  # Convert to g(x) <= 0 form
                    
                inequality_constraints.append(ineq_constr)
                
            elif '=' in constraint and not constraint.startswith('obj'):
                parts = constraint.split('=')
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                
                def eq_constr(x, l=lhs, r=rhs):
                    # Replace variables in expression
                    expr = l
                    for var_name, idx in var_map.items():
                        expr = expr.replace(var_name, f'x[{idx}]')
                    expr = expr.replace('^', '**')
                    lhs_val = eval(expr)
                    
                    # Evaluate right hand side
                    expr = r
                    for var_name, idx in var_map.items():
                        expr = expr.replace(var_name, f'x[{idx}]')
                    expr = expr.replace('^', '**')
                    rhs_val = eval(expr)
                    
                    return lhs_val - rhs_val  # Convert to h(x) = 0 form
                    
                equality_constraints.append(eq_constr)
                
        return {
            'x0': x0,
            'f': objective_func,
            'c_ineq': inequality_constraints,
            'c_eq': equality_constraints,
            'bounds': bounds
        }


# Example usage
def test_apm_parser():
    # Example APM content from your message
    apm_content = """
Model hs04
  Variables
    x[1] = 1.125, >= 1
    x[2] = 0.125, >= 0
    obj
  End Variables

  Equations
    ! best known objective = 8/3
    obj = (x[1]+1)^3/3 + x[2]
  End Equations
End Model
"""
    
    parser = APMParser()
    problem = parser.parse(apm_content)
    
    print("Model parsed successfully!")
    print(f"Initial point: {problem['x0']}")
    print(f"Bounds: {problem['bounds']}")
    print(f"Objective at x0: {problem['f'](problem['x0'])}")
    
    # Test constraints
    for i, constr in enumerate(problem['c_ineq']):
        print(f"Inequality constraint {i}: {constr(problem['x0'])}")
    
    for i, constr in enumerate(problem['c_eq']):
        print(f"Equality constraint {i}: {constr(problem['x0'])}")
    
    return problem


if __name__ == "__main__":
    test_apm_parser()