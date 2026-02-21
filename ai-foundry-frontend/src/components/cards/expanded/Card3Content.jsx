import { motion } from 'framer-motion'

function Card3Content({ contentData, generatedAssets }) {
  const handleInstagramPost = () => {
    alert('Instagram post would be generated/scheduled (simulated).')
  }

  const handleTwitterPost = () => {
    alert('Twitter post would be generated/scheduled (simulated).')
  }

  /* Small inline SVG icon components for action buttons */
  const IconWrapper = ({ children, title }) => (
    <button className="bg-transparent border-none cursor-pointer p-0 flex items-center text-inherit transition-colors hover:text-[#1d9bf0]" title={title} aria-label={title} type="button">{children}</button>
  )

  const HeartIcon = ({ filled=false }) => (
    <svg viewBox="0 0 24 24" width="20" height="20" fill={filled ? '#e0245e' : 'none'} stroke={filled ? '#e0245e' : 'currentColor'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M20.8 4.6c-1.9-1.8-5-1.7-6.9.2l-.9.9-.9-.9C10.2 2.9 7.1 2.8 5.2 4.6 2.9 6.8 3 10.6 5.4 13.1L12 19.6l6.6-6.5c2.4-2.4 2.5-6.2.2-8.5z" />
    </svg>
  )

  const CommentIcon = () => (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  )

  const ShareIcon = () => (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M4 12v7a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-7" />
      <polyline points="16 6 12 2 8 6" />
      <line x1="12" y1="2" x2="12" y2="15" />
    </svg>
  )

  const BookmarkIcon = () => (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
    </svg>
  )

  const RetweetIcon = () => (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <polyline points="23 7 23 1 17 1" />
      <path d="M20 8v6a3 3 0 0 1-3 3H7" />
      <polyline points="1 17 1 23 7 23" />
      <path d="M4 16v-6a3 3 0 0 1 3-3h10" />
    </svg>
  )

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="space-y-6 p-6">
      <h2 className="text-4xl font-bold text-white mb-6">Generate and automate Instagram & Twitter posts</h2>
      
      {contentData && contentData.social_posts && contentData.social_posts.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-12 justify-items-center">
          {contentData.social_posts.map((post, index) => {
            const platform = (post.platform || '').toLowerCase()
            const isTwitter = platform.includes('twitter') || platform.includes('x')
            const isInstagram = platform.includes('instagram') || platform.includes('insta')

            if (isInstagram) {
              return (
                <div key={index} className="bg-black text-white border border-white/20 rounded-xl overflow-hidden shadow-[0_4px_12px_rgba(0,0,0,0.15)] transition-all duration-300 hover:translate-y-[-4px] hover:shadow-[0_8px_24px_rgba(0,0,0,0.2)] w-full">
                  <div className="flex justify-between items-center py-3.5 px-4 border-b border-white/20">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-[#f09433] via-[#dc2743] to-[#bc1888] border-2 border-black shadow-[0_0_0_1px_rgba(255,255,255,0.2)]" />
                      <div className="font-semibold text-sm text-white">YOURPAGENAME</div>
                    </div>
                    <div className="text-xl font-bold text-white cursor-pointer">⋯</div>
                  </div>

                  <div className="relative w-full aspect-square">
                    {(() => {
                      const imgUrl = generatedAssets?.[`post_${index + 1}_image_url`] || post.image_url || null
                      return (
                        <div className="w-full h-full flex items-center justify-center bg-black relative">
                          {imgUrl ? (
                            <>
                              <img
                                src={imgUrl}
                                alt={post.image_prompt || post.caption || 'Instagram image'}
                                className="w-full h-full object-cover"
                                onError={(e) => {
                                  e.target.onerror = null
                                  e.target.src = `https://placehold.co/800x800/CCCCCC/666666?text=Image+not+available`
                                }}
                              />
                              <div className="absolute inset-0 bg-black/70 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity duration-300 p-5">
                                <div className="text-white text-center text-sm leading-normal">
                                  <strong>Search Query:</strong> {post.image_prompt || '—'}
                                </div>
                              </div>
                            </>
                          ) : (
                            <div className="text-2xl text-white font-light tracking-[2px]">IMAGE</div>
                          )}
                        </div>
                      )
                    })()}
                  </div>

                  <div className="flex justify-between py-2 px-4">
                    <div className="flex gap-4">
                      <IconWrapper title="Like"><HeartIcon /></IconWrapper>
                      <IconWrapper title="Comment"><CommentIcon /></IconWrapper>
                      <IconWrapper title="Share"><ShareIcon /></IconWrapper>
                    </div>
                    <div className="flex gap-4">
                      <IconWrapper title="Save"><BookmarkIcon /></IconWrapper>
                    </div>
                  </div>

                  <div className="px-4 font-semibold text-sm text-white mb-2">{post.likes || post.like_count || '—'} likes</div>

                  <div className="px-4 text-sm text-white leading-normal mb-2">
                    <strong className="font-semibold mr-1.5">{post.handle || post.username || 'YOURPAGENAME'}</strong> {post.content || post.caption || ''}
                  </div>

                  <div className="px-4 text-sm text-white/60 mb-3 cursor-pointer">View all {post.comments_count || post.comment_count || '0'} comments</div>

                  <div className="px-4 py-4 border-t border-white/20">
                    <input placeholder="Add a comment..." className="w-full border-none outline-none text-sm text-white bg-transparent placeholder:text-white/60" />
                  </div>
                </div>
              )
            }

            if (isTwitter) {
              const imgUrl = generatedAssets?.[`post_${index + 1}_image_url`] || post.image_url || null
              const duration = post.duration || post.video_duration || post.video_time || ''
              return (
                <div key={index} className="bg-black text-white border border-white/20 rounded-xl overflow-hidden shadow-[0_4px_12px_rgba(0,0,0,0.15)] transition-all duration-300 hover:translate-y-[-4px] hover:shadow-[0_8px_24px_rgba(0,0,0,0.2)]">
                  <div className="flex justify-between items-start pt-3 px-4">
                    <div className="flex gap-3">
                      <div className="w-12 h-12 rounded-full bg-gray-300 bg-cover bg-center flex-shrink-0" style={{ backgroundImage: `url('${post.avatar_url || 'https://placehold.co/80x80/cccccc/666666'}')` }} />
                      <div className="flex flex-col">
                        <div className="flex items-center gap-1">
                          <span className="font-bold text-white text-[15px]">{post.handle || post.username || 'YOURPAGENAME'}</span>
                          {post.verified ? <span className="text-[#1d9bf0] text-base">✔︎</span> : null}
                        </div>
                        <div className="text-white/60 text-[13px]">{post.sub || post.time || ''}</div>
                      </div>
                    </div>
                    <div className="text-white cursor-pointer">⋯</div>
                  </div>

                  <div className="py-0 px-4 pb-3 text-white text-[15px] leading-normal whitespace-pre-wrap">{post.content}</div>

                  <div className="mx-4 mb-3 relative">
                    {imgUrl ? (
                      <div className="relative w-full overflow-hidden rounded-2xl bg-black">
                        <img
                          src={imgUrl}
                          alt={post.image_prompt || post.caption || 'Tweet media'}
                          className="w-full block"
                          onError={(e) => {
                            e.target.onerror = null
                            e.target.src = `https://placehold.co/800x400/CCCCCC/666666?text=Loading+Image...`
                          }}
                        />
                        {duration ? <div className="absolute bottom-2 right-2 bg-black/75 text-white py-0.5 px-1.5 rounded text-xs font-semibold">{duration}</div> : null}
                        <div className="absolute inset-0 bg-black/75 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity duration-300 p-5">
                          <div className="text-white text-center text-[13px] leading-normal">
                            <strong>Search Query:</strong> {post.image_prompt || '—'}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="w-full aspect-video bg-gradient-to-br from-[#667eea] to-[#764ba2] flex flex-col items-center justify-center p-5 gap-4 rounded-2xl">
                        <div className="w-10 h-10 border-4 border-white/30 border-t-white rounded-full animate-spin"></div>
                        <div className="text-white text-center text-[13px] leading-normal">
                          <strong>Generating image for:</strong> {post.image_prompt}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="py-0 px-4 pb-3 border-t border-white/20 pt-3">
                    <div className="flex justify-around gap-4">
                      <span className="flex items-center gap-2 text-white text-[13px] cursor-pointer transition-colors hover:text-[#1d9bf0]">
                        <span className="flex items-center"><IconWrapper title="Replies"><CommentIcon/></IconWrapper></span>
                        {post.replies || post.reply_count || '---'}
                      </span>

                      <span className="flex items-center gap-2 text-white text-[13px] cursor-pointer transition-colors hover:text-[#1d9bf0]">
                        <span className="flex items-center"><IconWrapper title="Retweets"><RetweetIcon/></IconWrapper></span>
                        {post.retweets || post.retweet_count || '---'}
                      </span>

                      <span className="flex items-center gap-2 text-white text-[13px] cursor-pointer transition-colors hover:text-[#1d9bf0]">
                        <span className="flex items-center"><IconWrapper title="Likes"><HeartIcon/></IconWrapper></span>
                        {post.likes || post.like_count || '---'}
                      </span>
                    </div>
                  </div>
                </div>
              )
            }

            return null
          })}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-black/40 rounded-2xl p-6 border border-white/10 text-white/90">
            <h3 className="font-semibold mb-2">Instagram</h3>
            <p className="mb-4">⏳ Waiting for Instagram posts to be generated...</p>
            <button onClick={handleInstagramPost} className="py-2 px-4 bg-pink-500 hover:bg-pink-600 rounded-md text-white transition-colors">Generate IG Post</button>
          </div>

          <div className="bg-black/40 rounded-2xl p-6 border border-white/10 text-white/90">
            <h3 className="font-semibold mb-2">Twitter / X</h3>
            <p className="mb-4">⏳ Waiting for tweets to be generated...</p>
            <button onClick={handleTwitterPost} className="py-2 px-4 bg-blue-500 hover:bg-blue-600 rounded-md text-white transition-colors">Generate Tweet</button>
          </div>
        </div>
      )}
    </motion.div>
  )
}

export default Card3Content
