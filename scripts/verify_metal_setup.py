#!/usr/bin/env python3
"""Verify Metal and Apple Silicon setup for Shvayambhu development.

This script checks for required tools and capabilities needed for
optimal performance on Apple Silicon devices.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MetalSetupVerifier:
    """Verifies Metal and Apple Silicon development setup."""
    
    def __init__(self):
        self.results = {
            "platform": {},
            "xcode": {},
            "metal": {},
            "mlx": {},
            "python": {},
            "memory": {},
            "recommendations": []
        }
    
    def run_command(self, cmd: List[str]) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except FileNotFoundError:
            return False, "Command not found"
        except Exception as e:
            return False, str(e)
    
    def check_platform(self) -> bool:
        """Check if running on Apple Silicon."""
        logger.info("Checking platform compatibility...")
        
        # Check if macOS
        if platform.system() != "Darwin":
            self.results["platform"]["os"] = platform.system()
            self.results["platform"]["compatible"] = False
            self.results["recommendations"].append(
                "Shvayambhu requires macOS for Metal acceleration"
            )
            return False
        
        # Check processor
        processor = platform.processor()
        machine = platform.machine()
        
        self.results["platform"]["os"] = "macOS"
        self.results["platform"]["version"] = platform.mac_ver()[0]
        self.results["platform"]["processor"] = processor
        self.results["platform"]["machine"] = machine
        
        # Check for Apple Silicon
        is_apple_silicon = machine == "arm64" or "Apple" in processor
        self.results["platform"]["apple_silicon"] = is_apple_silicon
        self.results["platform"]["compatible"] = is_apple_silicon
        
        if not is_apple_silicon:
            self.results["recommendations"].append(
                "Apple Silicon (M1/M2/M3) recommended for optimal performance"
            )
        
        logger.info(f"Platform: {platform.system()} {platform.mac_ver()[0]} on {machine}")
        return is_apple_silicon
    
    def check_xcode(self) -> bool:
        """Check Xcode and command line tools installation."""
        logger.info("Checking Xcode installation...")
        
        # Check xcode-select
        success, output = self.run_command(["xcode-select", "-p"])
        if not success:
            self.results["xcode"]["installed"] = False
            self.results["recommendations"].append(
                "Install Xcode command line tools: xcode-select --install"
            )
            return False
        
        self.results["xcode"]["path"] = output
        
        # Check Xcode version
        success, output = self.run_command(["xcodebuild", "-version"])
        if success and output:
            lines = output.split('\n')
            if lines:
                version_line = lines[0]
                self.results["xcode"]["version"] = version_line
                
                # Extract version number
                try:
                    version_parts = version_line.split()
                    if len(version_parts) >= 2:
                        version_num = float(version_parts[1])
                        self.results["xcode"]["version_number"] = version_num
                        
                        if version_num < 14.0:
                            self.results["recommendations"].append(
                                "Xcode 14+ recommended for Metal 3 support"
                            )
                except (ValueError, IndexError):
                    pass
        
        self.results["xcode"]["installed"] = True
        logger.info(f"Xcode found: {self.results['xcode'].get('version', 'Unknown version')}")
        return True
    
    def check_metal(self) -> bool:
        """Check Metal capabilities."""
        logger.info("Checking Metal capabilities...")
        
        try:
            # Try to import and use Metal through PyObjC
            import objc
            import Metal
            
            # Get default Metal device
            device = Metal.MTLCreateSystemDefaultDevice()
            if device is None:
                self.results["metal"]["available"] = False
                self.results["recommendations"].append(
                    "Metal device not available"
                )
                return False
            
            self.results["metal"]["available"] = True
            self.results["metal"]["device_name"] = str(device.name())
            
            # Check Metal version support
            if hasattr(device, 'supportsFamily_'):
                # Check for various Metal feature sets
                families = []
                
                # Metal 3 family support (Apple Silicon)
                if device.supportsFamily_(7):  # MTLGPUFamilyApple7 (M1)
                    families.append("Apple7 (M1)")
                if device.supportsFamily_(8):  # MTLGPUFamilyApple8 (M2)
                    families.append("Apple8 (M2)")
                if device.supportsFamily_(9):  # MTLGPUFamilyApple9 (M3)
                    families.append("Apple9 (M3)")
                
                self.results["metal"]["gpu_families"] = families
            
            # Check unified memory
            self.results["metal"]["unified_memory"] = device.hasUnifiedMemory()
            self.results["metal"]["max_threads_per_group"] = device.maxThreadsPerThreadgroup().width
            
            logger.info(f"Metal device: {device.name()}")
            logger.info(f"Unified memory: {device.hasUnifiedMemory()}")
            
            return True
            
        except ImportError:
            self.results["metal"]["available"] = False
            self.results["recommendations"].append(
                "PyObjC not available. Install with: pip install pyobjc-framework-Metal"
            )
            return False
        except Exception as e:
            self.results["metal"]["available"] = False
            self.results["metal"]["error"] = str(e)
            return False
    
    def check_mlx(self) -> bool:
        """Check MLX framework installation."""
        logger.info("Checking MLX framework...")
        
        try:
            import mlx
            import mlx.core as mx
            
            self.results["mlx"]["installed"] = True
            self.results["mlx"]["version"] = getattr(mlx, '__version__', 'Unknown')
            
            # Test basic MLX functionality
            try:
                # Simple test
                a = mx.array([1, 2, 3])
                b = mx.array([4, 5, 6])
                c = a + b
                mx.eval(c)
                
                self.results["mlx"]["functional"] = True
                logger.info(f"MLX {self.results['mlx']['version']} working correctly")
                
            except Exception as e:
                self.results["mlx"]["functional"] = False
                self.results["mlx"]["test_error"] = str(e)
                self.results["recommendations"].append(
                    f"MLX test failed: {e}"
                )
            
            return True
            
        except ImportError:
            self.results["mlx"]["installed"] = False
            self.results["recommendations"].append(
                "Install MLX framework: pip install mlx"
            )
            return False
    
    def check_python_environment(self) -> bool:
        """Check Python environment and key packages."""
        logger.info("Checking Python environment...")
        
        # Python version
        python_version = platform.python_version()
        self.results["python"]["version"] = python_version
        
        # Check if Python 3.11+
        version_parts = python_version.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        python_compatible = major == 3 and minor >= 11
        
        self.results["python"]["compatible"] = python_compatible
        if not python_compatible:
            self.results["recommendations"].append(
                "Python 3.11+ required for optimal performance"
            )
        
        # Check key packages
        packages = ["torch", "numpy", "scipy"]
        installed_packages = {}
        
        for package in packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                installed_packages[package] = version
            except ImportError:
                installed_packages[package] = None
                self.results["recommendations"].append(
                    f"Install {package}: pip install {package}"
                )
        
        self.results["python"]["packages"] = installed_packages
        
        logger.info(f"Python {python_version}")
        return python_compatible
    
    def check_memory_info(self) -> bool:
        """Check system memory information."""
        logger.info("Checking memory configuration...")
        
        try:
            # Get memory info using vm_stat
            success, output = self.run_command(["vm_stat"])
            if success:
                # Parse vm_stat output for memory info
                lines = output.split('\n')
                memory_info = {}
                
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip().replace('.', '')
                        if value.isdigit():
                            memory_info[key] = int(value)
                
                # Calculate total memory (pages * page_size)
                if 'Pages free' in memory_info and 'Pages active' in memory_info:
                    # Page size is typically 4096 bytes on Apple Silicon
                    page_size = 4096
                    total_pages = sum(memory_info.values())
                    total_memory_gb = (total_pages * page_size) / (1024**3)
                    
                    self.results["memory"]["total_gb"] = round(total_memory_gb, 1)
                    
                    # Memory recommendations
                    if total_memory_gb < 16:
                        self.results["recommendations"].append(
                            "16GB+ RAM recommended for 7B model training"
                        )
                    elif total_memory_gb < 32:
                        self.results["recommendations"].append(
                            "32GB+ RAM recommended for 13B model training"
                        )
            
            # Check swap usage
            success, output = self.run_command(["sysctl", "vm.swapusage"])
            if success and "used" in output:
                self.results["memory"]["swap_info"] = output
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return False
    
    def generate_report(self) -> str:
        """Generate a comprehensive setup report."""
        report = ["Shvayambhu Metal Setup Verification Report", "=" * 50, ""]
        
        # Platform info
        report.append("PLATFORM INFORMATION:")
        platform_info = self.results["platform"]
        report.append(f"  OS: {platform_info.get('os', 'Unknown')} {platform_info.get('version', '')}")
        report.append(f"  Architecture: {platform_info.get('machine', 'Unknown')}")
        report.append(f"  Apple Silicon: {'✅' if platform_info.get('apple_silicon') else '❌'}")
        report.append("")
        
        # Xcode info
        report.append("XCODE & DEVELOPMENT TOOLS:")
        xcode_info = self.results["xcode"]
        if xcode_info.get("installed"):
            report.append(f"  Xcode: ✅ {xcode_info.get('version', 'Unknown')}")
            report.append(f"  Path: {xcode_info.get('path', 'Unknown')}")
        else:
            report.append("  Xcode: ❌ Not installed")
        report.append("")
        
        # Metal info
        report.append("METAL CAPABILITIES:")
        metal_info = self.results["metal"]
        if metal_info.get("available"):
            report.append(f"  Metal: ✅ Available")
            report.append(f"  Device: {metal_info.get('device_name', 'Unknown')}")
            report.append(f"  Unified Memory: {'✅' if metal_info.get('unified_memory') else '❌'}")
            if metal_info.get("gpu_families"):
                report.append(f"  GPU Families: {', '.join(metal_info['gpu_families'])}")
        else:
            report.append("  Metal: ❌ Not available")
        report.append("")
        
        # MLX info
        report.append("MLX FRAMEWORK:")
        mlx_info = self.results["mlx"]
        if mlx_info.get("installed"):
            report.append(f"  MLX: ✅ v{mlx_info.get('version', 'Unknown')}")
            if mlx_info.get("functional"):
                report.append("  Functionality: ✅ Working")
            else:
                report.append("  Functionality: ❌ Issues detected")
        else:
            report.append("  MLX: ❌ Not installed")
        report.append("")
        
        # Python info
        report.append("PYTHON ENVIRONMENT:")
        python_info = self.results["python"]
        report.append(f"  Python: {'✅' if python_info.get('compatible') else '❌'} v{python_info.get('version')}")
        
        packages = python_info.get("packages", {})
        for pkg, version in packages.items():
            status = "✅" if version else "❌"
            version_str = f"v{version}" if version else "Not installed"
            report.append(f"  {pkg}: {status} {version_str}")
        report.append("")
        
        # Memory info
        report.append("MEMORY CONFIGURATION:")
        memory_info = self.results["memory"]
        if memory_info.get("total_gb"):
            total_gb = memory_info["total_gb"]
            report.append(f"  Total RAM: {total_gb} GB")
            
            # Memory status
            if total_gb >= 32:
                report.append("  Memory Status: ✅ Excellent (13B+ models)")
            elif total_gb >= 16:
                report.append("  Memory Status: ✅ Good (7B models)")
            elif total_gb >= 8:
                report.append("  Memory Status: ⚠️  Minimum (small models)")
            else:
                report.append("  Memory Status: ❌ Insufficient")
        report.append("")
        
        # Recommendations
        if self.results["recommendations"]:
            report.append("RECOMMENDATIONS:")
            for rec in self.results["recommendations"]:
                report.append(f"  • {rec}")
            report.append("")
        
        # Overall status
        overall_good = (
            self.results["platform"].get("apple_silicon", False) and
            self.results["xcode"].get("installed", False) and
            self.results["metal"].get("available", False) and
            self.results["python"].get("compatible", False)
        )
        
        report.append("OVERALL STATUS:")
        if overall_good:
            report.append("  ✅ System ready for Shvayambhu development")
        else:
            report.append("  ❌ System needs additional setup")
        
        return '\n'.join(report)
    
    def save_report(self, filepath: str):
        """Save detailed results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def run_all_checks(self) -> bool:
        """Run all verification checks."""
        logger.info("Starting Metal setup verification...")
        
        checks = [
            self.check_platform(),
            self.check_xcode(),
            self.check_metal(),
            self.check_mlx(),
            self.check_python_environment(),
            self.check_memory_info()
        ]
        
        return all(checks)


def main():
    """Main entry point."""
    verifier = MetalSetupVerifier()
    
    # Run all checks
    success = verifier.run_all_checks()
    
    # Generate and print report
    report = verifier.generate_report()
    print(report)
    
    # Save detailed results
    results_path = "metal_setup_results.json"
    verifier.save_report(results_path)
    logger.info(f"Detailed results saved to: {results_path}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()