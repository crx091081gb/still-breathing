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
    //const cv::Rect& _track;
    
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

class sortableVec {
    public:
        ofVec3f v;
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
    ofParameter<float> gridMinArea, gridMaxArea, gridThreshold;
    ofParameter<int> trackerPersistence, trackerMaxDistance, frameVelThresh, stillThresh;
    ofParameter<bool> invert;
    ofParameter<bool> live;
    
    vector<ofxCv::ContourFinder> contourFinders;
    vector<ofImage> scanSlices;
    
    cv::Mat colourImg;
    ofImage ofColourImg;
    
    cv::Mat depthImg;
    ofImage ofDepthImg;
    ofImage ofThreshImg;
    
    ofxCv::ContourFinder depthContourFinder;
    ofxCv::RectTrackerFollower<BlobPeople> tracker;
    std::map<int, uint64_t> stillLabels;
    std::map<int, cv::Rect> stillRects;
    
    const int rows = 4;
    const int cols = 4;
    const int chunks = 16;
    ofImage ofDepthGridTest;
    //cv::Mat grid;
    cv::Rect depthImgROIGrid[16];
    ofxCv::ContourFinder gridContours[16];
    
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
        bool sortPeople(BlobPeople &b1, BlobPeople &b2);
        bool sortVectorByLength(ofVec3f & a, ofVec3f & b);
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
