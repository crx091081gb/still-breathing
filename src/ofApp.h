#pragma once

#include "ofMain.h"
#include "ofxEvm.h"
#include "ofxGui.h"
#include "ofxOpenCv.h"
#include "ofxCv.h"
#include "ofxKinect.h"
#include "ofxVideoRecorder.h"

class BlobPeople : public ofxCv::RectFollower {
protected:
    ofColor color;
    ofVec2f cur;
    
public:
    ofVec2f smooth;
    int height;
    int bottom;
    int left;
    int right;
    //    BlobPeople(float &slitRatio);
    void setup(const cv::Rect& track);
    void update(const cv::Rect& track);
    //void kill();
    void draw();
};

class ofApp : public ofBaseApp{

    ofxKinect kinect;
    ofxEvm evm;
    ofVideoGrabber video;
    
    ofxPanel gui;
    ofxFloatSlider alpha;
    ofxFloatSlider cutoff;
    ofxFloatSlider min_freq;
    ofxFloatSlider max_freq;
    ofxFloatSlider chrome;
    ofParameter<float> minArea, maxArea, breathBlobThreshold, depthThreshold, minDepthArea, maxDepthArea, slitRatio;
    ofParameter<int> trackerPersistence, trackerMaxDistance;
    ofParameter<bool> invert;
    ofParameter<bool> live;
    
    vector<ofxCv::ContourFinder> contourFinders;
    vector<ofImage> scanSlices;
    
    cv::Mat colourImg;
    ofImage ofColourImg;
    
    cv::Mat depthImg;
    ofImage ofDepthImg;
    
    ofxCv::ContourFinder depthContourFinder;
    ofxCv::RectTrackerFollower<BlobPeople> tracker;
    
    ofxVideoRecorder vidRecorder1;
    ofxVideoRecorder vidRecorder2;
    
    ofVideoPlayer recordedVideoPlayback;
    ofVideoPlayer recordedDepthPlayback;
    
    string fileName1;
    string fileName2;
    string fileExt;
    
    bool bRecording;
    bool bPlaying;
    
    bool newFrame;
    
	public:
		void setup();
		void update();
		void draw();
        void exit();
        void scanSlice(ofPixels_<float> & src, ofPixels& dst, int offset);
        void updateParams();
        void recordingComplete(ofxVideoRecorderOutputFileCompleteEventArgs& args);
		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
};
