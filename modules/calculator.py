import torch

class Calculator:

    def get_calculator(calculator_name, model_name=None):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"using device {device}...")

        if calculator_name == "deepmd":

            """
            available models:
            - DPA3-v2-OpenLAM
            - DPA3-v2-MPtrj

            if not specified, DPA3-v2-OpenLAM
            """

            from deepmd.calculator import DP
            
            if model_name == 'DPA3-v2-OpenLAM' or not model_name:
                return DP(model='../../pretrained_models/dpa3-openlam.pth')
            else:
                return DP(model='../../pretrained_models/dpa3-mptrj.pth')

        elif calculator_name == "fair-chem":

            """
            available models:
            - eSEN-30M-OAM
            - eSEN-30M-MP
            - eqV2 M
            - eqV2 S DeNS

            if not specified, eqV2 M
            """

            from fairchem.core import OCPCalculator
            if not model_name: model_name = 'eqV2 M'
            return OCPCalculator(
                model_name=model_name,
                local_cache="pretrained_models",
                cpu=(device.type == 'cpu'),
            )
        
        elif calculator_name == "grace":

            """
            available models:
            - GRACE-2L-OAM
            - GRACE-1L-OAM
            - GRACE-2L-MPtrj
            if not specified, GRACE-2L-OAM
            """

            from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels
            if model_name == 'GRACE-1L-OAM': return grace_fm(GRACEModels.GRACE_1L_OAM)
            else: return grace_fm(GRACEModels.GRACE_2L_OAM)

        elif calculator_name == "mace":

            """
            available models:
            - MACE-MPA-0
            - MACE-MP-0
            
            if not specified, MACE-MPA-0
            """
            
            from mace.calculators import mace_mp
            if not model_name: model_name = "medium"
            
            # elif model_name == 'MACE-MP-0': model_name = 'https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model'
            
            return mace_mp(model='medium', device=device.type, default_dtype='float64')

        elif calculator_name == "mattersim":

            """
            available models: MatterSim-v1.0.0-5M
            """

            from mattersim.forcefield import MatterSimCalculator
            return MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)

        elif calculator_name == "orb":

            """"
            available models:
            - ORB
            - ORB MPTrj
            """

            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
            
            if model_name == 'orb-mptrj': model = pretrained.orb_mptraj_only_v2
            else: model = pretrained.orb_v2(device=device)
            
            return ORBCalculator(model, device=device)

        elif calculator_name == "sevenn":

            """"
            available models:
            - SevenNet-MF-ompa
            - SevenNet-l3i5
            """

            from sevenn.calculator import SevenNetCalculator
            if not model_name: model_name = '7net-mf-ompa'                
            return SevenNetCalculator(model=model_name, modal='mpa')

        else:
            raise ValueError("calculator not supported...")

    @staticmethod
    def help():
        """
        list of supported calculators and models
        last updated march 26th

        1. "deepmd": DPA3-v2-OpenLAM (default), DPA3-v2-MPtrj
        2. "fair-chem": eSEN-30M-OAM, eSEN-30M-MP, eqV2 M (default), eqV2 S DeNS
        3. "grace": GRACE-2L-OAM (default), GRACE-1L-OAM, GRACE-2L-MPtrj        
        4. "mace": MACE-MPA-0 (default), MACE-MP-0
        5. "mattersim": MatterSim-v1.0.0-5M
        6. "orb": ORB (default), ORB MPTrj
        7. "sevenn": SevenNet-MF-ompa (default), SevenNet-l3i5
        
        usage:
        > calc = Calculator.get_calculator("deepmd", model_name="DPA3-v2-MPtrj")
        > calc = Calculator.get_calculator("mace")        
        """
        print(Calculator.help.__doc__)
