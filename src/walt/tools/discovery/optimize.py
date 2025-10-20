#!/usr/bin/env python3
"""
tool optimization utilities.
Handles all optimization transformations for discovered tools.
"""
import os
from copy import deepcopy
import re
from typing import Dict, List, Any


def optimize_tool(tool) -> Dict[str, Any]:
    """
    Apply full optimization to tool.
    
    Args:
        tool: tool dict or Pydantic model
        
    Returns:
        Optimized tool dict
    """
    optimizer = toolOptimizer()
    return optimizer.optimize(tool)


class toolOptimizer:
    """Handles all tool optimization transformations."""
    
    def optimize(self, tool) -> Dict[str, Any]:
        """
        Apply full optimization to tool.
        
        Args:
            tool: tool dict or Pydantic model
            
        Returns:
            Optimized tool dict
        """
        # Convert tool to dict format if needed
        if hasattr(tool, "model_dump"):
            tool_dict = tool.model_dump()
        else:
            tool_dict = tool

        optimized = deepcopy(tool_dict)
        steps = tool_dict.get("steps", [])

        # Try URL operation optimization first
        if self._can_convert_to_url_operation(steps):
            optimized = self._convert_to_url_operation(optimized)
        else:
            # Remove defensive programming
            steps = self._remove_defensive_steps(steps)
            # Remove redundant navigation steps
            optimized["steps"] = self._remove_redundant_navigation(steps)

        # Optimize parameters
        optimized["input_schema"] = self._optimize_parameters(
            tool_dict.get("input_schema", [])
        )

        # Clean metadata
        if "tool_analysis" in optimized:
            del optimized["tool_analysis"]

        return optimized

    def _can_convert_to_url_operation(self, steps: List[Dict]) -> bool:
        """Check if tool can be converted to a direct URL operation."""
        if len(steps) < 3:
            return False

        # Look for navigation + form + submit pattern
        has_navigation = any(step.get("type") == "go_to_url" for step in steps[:2])
        has_form_fill = any(step.get("type") == "input_text" for step in steps)
        has_submit = any(
            step.get("type") in ["click_element", "key_press"] for step in steps
        )

        return has_navigation and has_form_fill and has_submit

    def _convert_to_url_operation(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert multi-step tool to single URL operation."""
        optimized = deepcopy(tool)
        steps = tool.get("steps", [])

        # Extract form parameters
        form_params = []
        for step in steps:
            if step.get("type") == "input_text":
                field_name = self._extract_field_name(step)
                if field_name:
                    form_params.append(f"{field_name}={{{{ {field_name} }}}}")

        # Create single URL operation
        base_url = None
        for step in steps:
            if step.get("type") == "go_to_url":
                base_url = step.get("url", "")
                break

        if base_url and form_params:
            query_string = "&".join(form_params)
            new_url = f"{base_url}?{query_string}"

            optimized["steps"] = [
                {
                    "type": "go_to_url",
                    "url": new_url,
                    "description": "Navigate directly with search parameters",
                }
            ]

        return optimized

    def _extract_field_name(self, step: Dict[str, Any]) -> str:
        """Extract field name from input step."""
        # Try CSS selector patterns
        css_selector = step.get("css_selector", "")
        
        # Extract name attribute
        name_match = re.search(r'name="([^"]+)"', css_selector)
        if name_match:
            return name_match.group(1)

        # Extract id attribute as fallback
        id_match = re.search(r'id="([^"]+)"', css_selector)
        if id_match:
            return id_match.group(1)

        return ""

    def _remove_defensive_steps(self, steps: List[Dict]) -> List[Dict]:
        """Remove defensive programming steps."""
        essential_steps = []

        for step in steps:
            # Skip verification steps
            description = step.get("description", "").lower()
            if any(
                word in description for word in ["verify", "analyze", "inspect", "confirm"]
            ):
                continue

            # Keep essential steps
            essential_steps.append(step)

        return essential_steps

    def _remove_redundant_navigation(self, steps: List[Dict]) -> List[Dict]:
        """Remove redundant navigation steps that don't add value."""
        if len(steps) <= 1:
            return steps
            
        optimized_steps = []
        
        for i, step in enumerate(steps):
            step_type = step.get("type")
            
            # Skip redundant initial navigation if:
            # 1. It's the first step
            # 2. It's a simple navigation to a base page
            # 3. The next step can work from any starting point
            if (i == 0 and 
                step_type == "navigation" and 
                self._is_redundant_initial_navigation(step, steps[1:] if len(steps) > 1 else [])):
                continue
                
            optimized_steps.append(step)
            
        return optimized_steps
    
    def _is_redundant_initial_navigation(self, nav_step: Dict, remaining_steps: List[Dict]) -> bool:
        """Check if an initial navigation step is redundant."""
        nav_url = nav_step.get("url", "")
        
        # CONSERVATIVE APPROACH: Only remove navigation if it's truly redundant
        # Most tools need their initial navigation to establish correct context
        
        # Only consider homepage navigation as potentially redundant
        redundant_patterns = [
            r"^[^/]*/?$",  # Just domain or domain with single slash (homepage)
        ]
        
        if any(re.search(pattern, nav_url) for pattern in redundant_patterns):
            # Even then, only if there are no URL operations that depend on current context
            if remaining_steps:
                for step in remaining_steps:
                    # If any step uses url_operation or {{current_url}}, keep the navigation
                    if (step.get("type") == "navigation" and 
                        (step.get("url_operation") is not None or 
                         "{{current_url}}" in str(step.get("url", "")))):
                        return False
                    # If any step references current context, keep the navigation
                    if "{{current_url}}" in str(step):
                        return False
                        
                # If no context-dependent steps, homepage navigation might be redundant
                return True
                    
        # All other navigation (including search pages, specific pages) is essential
        return False

    def _optimize_parameters(self, input_schema: List[Dict]) -> List[Dict]:
        """Optimize tool parameters by removing redundant ones."""
        optimized_params = []
        seen_names = set()

        for param in input_schema:
            name = param.get("name", "")
            if name and name not in seen_names:
                # Simplify parameter types
                param_type = param.get("type", "string")
                if param_type in ["text", "varchar", "char"]:
                    param["type"] = "string"

                optimized_params.append(param)
                seen_names.add(name)

        return optimized_params


async def optimize_existing_tools(args) -> int:
    """Optimize all existing base tools in the output directory."""
    import glob
    import json
    from .utils import FileManager
    
    tool_files = glob.glob(os.path.join(args.output_dir, "*", "*.tool.json"))
    
    if len(tool_files) == 0:
        print("❌ No existing base tools found to optimize")
        return 0
    
    print(f"⚡ Found {len(tool_files)} base tools to optimize")
    
    successful_optimizations = 0
    
    for tool_file in tool_files:
        tool_dir = os.path.dirname(tool_file)
        tool_name = os.path.basename(tool_file).replace(".tool.json", "")
        optimized_file = os.path.join(tool_dir, f"{tool_name}.optimized.json")
        
        # Skip if optimized version already exists (unless force regenerate)
        if os.path.exists(optimized_file) and not getattr(args, "force_regenerate", False):
            print(f"  ⏭️ {tool_name} already has optimized version, skipping")
            continue
        
        print(f"  ⚡ Optimizing tool: {tool_name}")
        
        try:
            # Load base tool
            with open(tool_file, "r") as f:
                base_tool = json.load(f)
            
            # Optimize it
            optimized_tool = optimize_tool(base_tool)
            
            # Save optimized version
            FileManager.save_tool_json(optimized_tool, optimized_file)
            
            print(f"  ✅ {tool_name} optimized successfully")
            successful_optimizations += 1
            
        except Exception as e:
            print(f"  ❌ Failed to optimize {tool_name}: {e}")
    
    print(f"⚡ Optimization complete: {successful_optimizations}/{len(tool_files)} tools optimized")
    return successful_optimizations

