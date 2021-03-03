###import cv2

#cap = cv2.VideoCapture("MA LIBELLULE.mp4")
#ret, frame = cap.read()
#while(1):
 #  ret, frame = cap.read()
 #  cv2.imshow('frame',frame)
  # if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
  #     cap.release()
  #     cv2.destroyAllWindows()
  #     break
  # cv2.imshow('frame',frame)


import vlc
Instance = vlc.Instance()
player = Instance.media_player_new()
Media = Instance.media_new('SomethingFromNothing.mkv')
player.set_media(Media)
player.play()