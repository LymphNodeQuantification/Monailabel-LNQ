"""
exec(open("/mnt/data/lnq-timc-diamonds.py").read())
"""

segDir = "/mnt/lnq-timc-batch1/measurements-segmentations"
ctDir = "/mnt/lnq-timc-batch1/ct-series"
axialBulgeRatio = .25

import fnmatch
import json
import numpy
import os

import DICOMLib

jsonPath = "/mnt/lnq-timc-batch1/measurements-anonymized.json"

measurements = json.loads(open(jsonPath).read())
measurementsByStudyUID = {}
for measurement in measurements:
    if measurement['StudyInstanceUID'] not in measurementsByStudyUID:
        measurementsByStudyUID[measurement['StudyInstanceUID']] = []
    measurementsByStudyUID[measurement['StudyInstanceUID']].append(measurement)

db = slicer.dicomDatabase
segLogic = slicer.modules.segmentations.logic()

failedMeasurements = []

for patient in db.patients():
    print(patient)
    for study in db.studiesForPatient(patient):
        print(study)
        seriesUID = measurementsByStudyUID[study][0]['SeriesInstanceUID'] # all measurements are on same ct series
        labelPath = f"{segDir}/{seriesUID}.nii.gz"
        ctPath = f"{ctDir}/{seriesUID}.nii.gz"
        if os.path.exists(labelPath) and os.path.exists(ctPath):
            print(f"Skipping {seriesUID}")
            continue
        ctNode = None
        slicer.mrmlScene.Clear()
        loadedNodeIDs = DICOMLib.loadSeriesByUID(list(db.seriesForStudy(study)))
        for nodeID in loadedNodeIDs:
            node = slicer.mrmlScene.GetNodeByID(nodeID)
            if node.IsA("vtkMRMLScalarVolumeNode"):
                ctNode = node
        markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "markups")
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "model")
        m2mNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsToModelNode", "m2m")
        m2mNode.SetAndObserveInputNodeID(markupsNode.GetID())
        m2mNode.SetAndObserveOutputModelNodeID(modelNode.GetID())
        lnSeg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "lnSeg")
        lnSeg.CreateDefaultDisplayNodes()
        lnSeg.SetReferenceImageGeometryParameterFromVolumeNode(ctNode)
        for measurement in measurementsByStudyUID[study]:
            try:
                if fnmatch.fnmatch(measurement['Location'], "*Lymph*"):
                    points = []
                    sumDistance = 0
                    for nodeID in loadedNodeIDs:
                        node = slicer.mrmlScene.GetNodeByID(nodeID)
                        if node.GetAttribute("DICOM.instanceUIDs") == measurement['SRSOPInstanceUID']:
                            point = [0,]*3
                            node.GetNthControlPointPosition(0, point)
                            points.append(point)
                            point = [0,]*3
                            node.GetNthControlPointPosition(1, point)
                            points.append(point)
                            sumDistance += node.GetLineLengthWorld()
                    markupsNode.RemoveAllControlPoints()
                    for point in points:
                        markupsNode.AddControlPoint(point)
                    center = numpy.array(points).mean(axis=0)
                    axialOffset = axialBulgeRatio * sumDistance / 2
                    center[2] += axialOffset
                    markupsNode.AddControlPoint(center)
                    center[2] -= 2 * axialOffset
                    markupsNode.AddControlPoint(center)
                    segLogic.ImportModelToSegmentationNode(modelNode, lnSeg)
            except IndexError:
                failedMeasurements.append(measurement)
        lnLabel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "lnLabel")
        segLogic.ExportAllSegmentsToLabelmapNode(lnSeg, lnLabel, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
        slicer.app.processEvents(qt.QEventLoop.ExcludeUserInputEvents)
        slicer.util.saveNode(lnLabel, f"{segDir}/{seriesUID}.nii.gz")
        slicer.util.saveNode(ctNode, f"{ctDir}/{seriesUID}.nii.gz")
        #break
    #break

print(f"Finished.  {len(failedMeasurements)} failed of {len(measurements)} total")

"""

            slicer.app.processEvents(qt.QEventLoop.ExcludeUserInputEvents)

for patient in db.patients():
    for study in db.studiesForPatient(patient):
        for series in db.seriesForStudy(study):
            fileLists = [db.filesForSeries(series)]
            selectedPlugins = slicer.modules.dicomPlugins.keys()
            messages = []
            loadablesByPlugin, loadEnabled = DICOMLib.getLoadablesFromFileLists(fileLists, selectedPlugins, messages,
                                                                                lambda progressLabel, progressValue, progressDialog=progressDialog: progressCallback(progressDialog, progressLabel, progressValue),
                                                                                self.pluginInstances)
"""
