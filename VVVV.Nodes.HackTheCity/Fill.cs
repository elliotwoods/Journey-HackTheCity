#region using
using System.Collections.Generic;
using System.Drawing;
using VVVV.Nodes.OpenCV;

using VVVV.PluginInterfaces.V2;
using VVVV.Utils.VMath;
using System;
using VVVV.Utils.VColor;
using Emgu.CV;

#endregion

namespace VVVV.Nodes.KC.HackTheCity
{
    public class FillInstance : IFilterInstance
    {
        public int Steps = 1;
        public int Seeds = 1;

        CVImage FMask = new CVImage();
        CVImage FUnmasked = new CVImage();

        StructuringElementEx kernel = new StructuringElementEx(5, 3, 3, 1, Emgu.CV.CvEnum.CV_ELEMENT_SHAPE.CV_SHAPE_ELLIPSE);
        Random RNG = new Random();
        public override void Allocate()
        {
            FOutput.Image.Initialise(FInput.Image.ImageAttributes.Size, TColorFormat.L8);
            FMask.Initialise(FInput.Image.ImageAttributes.Size, TColorFormat.L8);
            FUnmasked.Initialise(FInput.Image.ImageAttributes.Size, TColorFormat.L8);
        }

        public override void Process()
        {
            FInput.GetImage(FMask);
            Reset();
        }

        private unsafe void AddNoise()
        {
            byte* data = (byte*)FOutput.Image.Data;
            byte* mask = (byte*)FMask.Data;

            for (int i = 0; i < Seeds; i++)
            {
                int index = RNG.Next(0, FOutput.Image.Width * FOutput.Image.Height);
                if (mask[index] != 0)
                    data[index] = 255;
            }
        }

        public void Step()
        {
            CvInvoke.cvDilate(FOutput.CvMat, FUnmasked.CvMat, IntPtr.Zero, Steps);
            CvInvoke.cvCopy(FUnmasked.CvMat, FOutput.CvMat, FMask.CvMat);
            FOutput.Send();
            AddNoise();
        }

        public void Reset()
        {
            FOutput.Image.SetPixels(0.0);
            AddNoise(); 
        }
    }

    #region PluginInfo
    [PluginInfo(Name = "Fill", Category = "HackTheCity", Version = "", Help = "Inflate features in image, i.e. grow noise", Author = "elliotwoods", Credits = "", Tags = "")]
    #endregion PluginInfo
    public class FillNode : IFilterNode<FillInstance>
    {
        [Input("Steps", MinValue = 0, MaxValue = 64, DefaultValue = 1, IsSingle=true)]
        IDiffSpread<int> FIterations;

        [Input("Seeds per frame", MinValue = 0, MaxValue = 64, DefaultValue = 1, IsSingle = true)]
        IDiffSpread<int> FSeeds;

        [Input("Reset", IsBang = true)]
        ISpread<bool> FReset;

        protected override void Update(int InstanceCount, bool SpreadChanged)
        {
            foreach (var proc in FProcessor)
                proc.Step();

            if (FIterations.IsChanged)
                foreach (var proc in FProcessor)
                    proc.Steps = FIterations[0];

            if (FSeeds.IsChanged)
                foreach (var proc in FProcessor)
                    proc.Seeds = FSeeds[0];

            if (FReset[0])
                foreach (var proc in FProcessor)
                    proc.Reset();
        }
    }
}
